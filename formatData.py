import pandas as pd
import json
import exchange_calendars
from datetime import datetime

def get_tau_in_year(start: datetime, end: datetime):
    calendar = exchange_calendars.get_calendar("XHKG")
    days_in_2025 = len(calendar.schedule.loc[start:"2026-01-01"])
    if end.year == 2025:
        return len(calendar.schedule.loc[start:end]) / 252
    else:
        months = (end.year - 2026) * 12 + end.month 
        tau = days_in_2025 / 252 + months / 12
        return tau
    

if __name__ == "__main__":
    rawData = pd.read_csv('Data/20250901.csv')
    
    count = 0
    optiondict = {'HSI': 25617.42, 'num': None, 'rf': .03}
    
    for i, row in rawData.iterrows():
        if not row['series'].startswith("HSI"):
            continue
        if len(row['series']) != 10:
            continue
        
        code = row['series']
        strike = int(code[3:8])
        letter = code[8]
        letter = ord(letter) & 0xbf
        optionType = 'call' if letter / 12 <= 1 else 'put'
        month = (letter - 1) % 12 + 1
        year = int(code[-1]) + 2020 if code[-1] != '0' else 2030

        maturity = f"{year}-{month:02d}"

        try:
            _ = optiondict[maturity]
        except KeyError:
            optiondict[maturity] = {'strike': {'call': [], 'put': []},
                                    'price': {'call': [], 'put': []},
                                    'IV': {'call': [], 'put': []},
                                    'tau': None}
        optiondict[maturity]['strike'][optionType].append(strike)
        optiondict[maturity]['price'][optionType].append(row['settle'])
        optiondict[maturity]['IV'][optionType].append(row['IV'])
        
        if optiondict[maturity]['tau'] == None:
            optiondict[maturity]['tau'] = get_tau_in_year(datetime(2025, 9, 1), datetime(year, month, 29))
        count += 1
    optiondict['num'] = count
    print(f"{count} HSI option is found!")
    with open('Data/250901.json', 'w', encoding='utf-8') as f:
        json.dump(optiondict, f)
    print("json has ben written")