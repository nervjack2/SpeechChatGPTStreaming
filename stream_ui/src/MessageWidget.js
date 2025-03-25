import React from 'react';
import { Paper, Typography } from '@mui/material';

const styles = {
  user_text: {
    backgroundColor: '#add8e6',
    padding: '8px',
    borderRadius: '10px',
    margin: '10px',
    textAlign: 'right',
  },
  system_text: {
    backgroundColor: '#d3d3d3',
    padding: '8px',
    borderRadius: '10px',
    margin: '10px',
    textAlign: 'left',
  },
};

function MessageWidget({ text, type }) {
  return (
    <Paper style={styles[type]}>
      <Typography variant="body1">{text}</Typography>
    </Paper>
  );
}

export default MessageWidget;