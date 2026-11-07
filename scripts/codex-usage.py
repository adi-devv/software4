"""Live Codex usage limits for all accounts. Read-only; uses no model tokens.

Run:  python scripts/codex-usage.py        (refreshes every 30 s, Ctrl+C to stop)
      python scripts/codex-usage.py --once
"""
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

HOMES = {'7859': '.codex-7859', 'w': '.codex-w', 'amit': '.codex-amit'}
URL = 'https://chatgpt.com/backend-api/wham/usage'
IST = timezone(timedelta(hours=5, minutes=30))
EVERY = 30


def fetch(home):
    path = os.path.join(os.path.expanduser('~'), home, 'auth.json')
    tok = json.load(open(path))['tokens']
    req = urllib.request.Request(URL, headers={
        'Authorization': 'Bearer ' + tok['access_token'],
        'chatgpt-account-id': tok.get('account_id', ''),
        'User-Agent': 'codex-cli'})
    return json.load(urllib.request.urlopen(req, timeout=15))


def render():
    now = datetime.now(IST).strftime('%I:%M:%S %p').lstrip('0')
    lines = ['Codex  %s IST' % now, '']
    for name, home in HOMES.items():
        try:
            rl = fetch(home)['rate_limit']
            p, w = rl['primary_window'], rl['secondary_window']
            at = datetime.fromtimestamp(p['reset_at'], IST).strftime('%I:%M %p').lstrip('0')
            lines.append('%-5s 5h %3d%%  week %3d%%   resets %s' % (
                name, 100 - p['used_percent'], 100 - w['used_percent'], at))
        except Exception as e:
            lines.append('%-5s unavailable' % name)
    return '\n'.join(lines)


if __name__ == '__main__':
    if '--once' in sys.argv:
        print(render())
        sys.exit(0)
    try:
        while True:
            out = render()
            os.system('cls' if os.name == 'nt' else 'clear')
            print(out, flush=True)
            time.sleep(EVERY)
    except KeyboardInterrupt:
        pass
