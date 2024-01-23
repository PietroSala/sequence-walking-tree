from datetime import  timedelta
import pandas as pd


def slicer(vts, vte, stride, ow, ww, pw, granularity):
    # Dictionary
    granularity_map = {
        'days': 'days',
        'hours': 'hours',
        'minutes': 'minutes',
        'seconds': 'seconds'
    }

    # Calculate the timedelta based on the selected granularity
    delta_unit = granularity_map[granularity]
    delta = timedelta(**{delta_unit: stride})

    ow_delta = timedelta(**{delta_unit: ow})
    ww_delta = timedelta(**{delta_unit: ww})
    pw_delta = timedelta(**{delta_unit: pw})

    result = []
    current_start = vts

    while current_start + ow_delta + ww_delta + pw_delta <= vte:
        st = current_start
        ow = current_start + ow_delta
        ww = current_start + ow_delta + ww_delta
        pw = current_start + ow_delta + ww_delta + pw_delta
        result.append((st,ow, ww, pw))
        current_start += delta
    df = pd.DataFrame(result, columns=['st','ow', 'ww', 'pw'])
    df['i'] = df.index
    return df