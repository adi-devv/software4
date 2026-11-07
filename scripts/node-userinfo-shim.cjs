// Preload for the Codex Windows sandbox, where os.userInfo() throws ENOMEM (uv_os_get_passwd).
const os = require('os');
const real = os.userInfo;
os.userInfo = function userInfo(...args) {
  try {
    return real.apply(this, args);
  } catch {
    const username = process.env.USERNAME || process.env.USER || 'sandbox';
    return { uid: -1, gid: -1, username, homedir: os.homedir(), shell: null };
  }
};
