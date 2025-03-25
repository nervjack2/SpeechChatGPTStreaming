import torch
from .meldataset import mel_spectrogram as mel_fn

MAX_WAV_VALUE = 32768.0

def std_normal(size, device):
    '''
    Generate the standard Gaussian variable of a certain size
    '''
    return torch.normal(0, 1, size=size).to(device)


def sampling_given_noise_schedule_ddim(h, eval_n_sch, net, km, ref):
    alpha_infer, beta_infer, sigma_infer, steps_infer = eval_n_sch
    N = len(steps_infer)

    # prepare ref
    ref = mel_fn(ref, h.n_fft, h.num_mels, h.sampling_rate,
                 h.hop_size, h.win_size, h.fmin, h.fmax)
    ref = (ref - h.mel_m) / h.mel_s

    # prepare x_T
    L_mel = int(km.size(1) * h.sampling_rate / h.hop_size / h.km_rate)
    x_t = std_normal((1, h.num_mels, L_mel), km.device)

    # precompute, main net
    c, ref = net.URE(km, ref)

    alpha_prev = torch.cat([alpha_infer[0:1], alpha_infer[:-1]])

    for n in range(N - 1, -1, -1):
        ts = steps_infer[n] * torch.ones((1, 1)).to(km.device)
        eps = net(x_t, ts, c, ref)
        # DDIM update
        x0_t = (x_t - torch.sqrt(1 - alpha_infer[n] ** 2) * eps) / alpha_infer[n]
        if n > 0:
            x_t = alpha_prev[n] * x0_t + torch.sqrt(1 - alpha_prev[n] ** 2) * eps
        else:
            x_t = x0_t
    x = x_t * h.mel_s + h.mel_m

    return x

def compute_hyperparams_given_schedule(beta):
    '''
    Compute diffusion process hyperparameters

    Parameters:
    beta (tensor):  beta schedule

    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), beta/alpha (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    '''

    T = len(beta)
    alpha = 1 - beta
    sigma = beta + 0
    for t in range(1, T):
        alpha[t] *= alpha[t - 1]  # \alpha^2_t = \prod_{s=1}^t (1-\beta_s)
    alpha = torch.sqrt(alpha).to(beta.device)

    _dh = {}
    _dh['T'], _dh['beta'], _dh['alpha'] = T, beta, alpha
    return _dh


def map_noise_scale_to_time_step(alpha_infer, alpha):
    if alpha_infer < alpha[-1]:
        return len(alpha) - 1
    if alpha_infer > alpha[0]:
        return 0
    for t in range(len(alpha) - 1):
        if alpha[t + 1] <= alpha_infer <= alpha[t]:
            step_diff = alpha[t] - alpha_infer
            step_diff /= alpha[t] - alpha[t + 1]
            return t + step_diff.item()
    return -1


def get_eval_noise_schedule(N, dh, device):
    if N == 1000:
        noise_schedule = torch.linspace(0.000001, 0.01, 1000)
    elif N == 200:
        noise_schedule = torch.linspace(0.0001, 0.02, 200)
    elif N == 50:
        noise_schedule = torch.linspace(0.0001, 0.05, 50)
    elif N == 8:
        noise_schedule = torch.FloatTensor([
            6.689325005027058e-07, 1.0033881153503899e-05, 0.00015496854030061513,
            0.002387222135439515, 0.035597629845142365, 0.3681158423423767, 0.4735414385795593, 0.5
        ])
    elif N == 6:
        noise_schedule = torch.FloatTensor([
            1.7838445955931093e-06, 2.7984189728158526e-05, 0.00043231004383414984,
            0.006634317338466644, 0.09357017278671265, 0.6000000238418579
        ])
    elif N == 4:
        noise_schedule = torch.FloatTensor([
            3.2176e-04, 2.5743e-03, 2.5376e-02, 7.0414e-01
        ])
    elif N == 3:
        noise_schedule = torch.FloatTensor([
            9.0000e-05, 9.0000e-03, 6.0000e-01
        ])
    else:
        raise NotImplementedError

    T, alpha = dh['T'], dh['alpha']
    assert len(alpha) == T

    beta_infer = noise_schedule.to(device)
    N = len(beta_infer)
    alpha_infer = 1 - beta_infer
    sigma_infer = beta_infer + 0
    for n in range(1, N):
        alpha_infer[n] *= alpha_infer[n - 1]
        sigma_infer[n] *= (1 - alpha_infer[n - 1]) / (1 - alpha_infer[n])
    alpha_infer = torch.sqrt(alpha_infer)
    sigma_infer = torch.sqrt(sigma_infer)

    # mapping noise scales to time steps
    steps_infer = []
    for n in range(N):
        step = map_noise_scale_to_time_step(alpha_infer[n], alpha)
        if step >= 0:
            steps_infer.append(step)
    steps_infer = torch.FloatTensor(steps_infer).to(device)

    return alpha_infer, beta_infer, sigma_infer, steps_infer


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
