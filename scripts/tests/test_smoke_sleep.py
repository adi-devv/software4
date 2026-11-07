from scripts.smoke_sleep import summarize_scores


def test_retention_table_excludes_day_scores():
    rows = [dict(type='score', arm='ours', cycle=0, phase='day', skill='rule:A', correct=0)
            for _ in range(24)]
    rows += [dict(type='score', arm='ours', cycle=0, phase='hidden', skill='rule:A', correct=1)
             for _ in range(8)]
    rows += [dict(type='score', arm='ours', cycle=0, phase='canary', skill='canary:smoke', correct=1)]
    result = summarize_scores(rows)['ours']['0']
    assert result['rule:A'] == 1.0 and result['canary:smoke'] == 1.0
    assert result['rule:B'] is None
