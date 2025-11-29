import pandas as pd

sub = pd.read_csv('d:/novartis/novartis_ml_project/outputs/submission.csv')

print('='*80)
print('FINAL SUBMISSION VERIFICATION')
print('='*80)

print(f'\nShape: {sub.shape}')
print(f'\nColumns: {sub.columns.tolist()}')

print(f'\nVolume Statistics:')
print(sub['volume'].describe())

print(f'\nAny NaN values: {sub["volume"].isna().sum()}')
print(f'Any negative values: {(sub["volume"] < 0).sum()}')

print(f'\nSample predictions (first brand, all 24 months):')
brand1 = sub.iloc[:24]
print(brand1.to_string(index=False))

print(f'\n\nScenario breakdown:')
scenario1 = sub[sub['months_postgx'] == 0]
scenario2 = sub[sub['months_postgx'] == 6]
print(f'Scenario 1 brands (starting at month 0): {len(scenario1)}')
print(f'Scenario 2 brands (starting at month 6): {len(scenario2)}')
