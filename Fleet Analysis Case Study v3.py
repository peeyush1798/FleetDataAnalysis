#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Building a flexible function which is not range bound

import os
import pandas as pd
import re

def load_and_prepare_fleet_data(data_dir):
    all_data = []
    pattern = re.compile(r"Fleet_Data_(\d{4})\.csv")

    for file in os.listdir(data_dir):
        match = pattern.match(file)
        if match:
            year = int(match.group(1))
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path)

            # Cleaning and normalizing columns
            df.columns = df.columns.str.strip().str.lower()

            # filtering "type swap" from status
            if 'status' in df.columns:
                df = df[~df['status'].str.lower().str.contains("type swap", na=False)]

            # Adding an year column in each file before merging the dataset
            df["year"] = year
            all_data.append(df)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        raise FileNotFoundError("No fleet files found.")


# In[3]:


# PWD
import os
os.getcwd()


# In[4]:


df_fleet = load_and_prepare_fleet_data(r"C:\Users\Peeyush.sharma\Fleet Data")


# In[5]:


df_fleet.head(25)


# In[6]:


# droping columns
df_fleet.drop(['number of seats estimated', 'number of seats'], axis=1, inplace=True)


# In[7]:


df_fleet.head(25)


# In[9]:


df_fleet.tail(25)


# In[10]:


df_fleet['year'].value_counts().sort_index()


# In[11]:


df_fleet.columns


# In[12]:


# Checking Type Swap Values
df_fleet[df_fleet['status'].str.contains("type swap", case=False, na=False)]


# In[13]:


# checking null values
df_fleet[df_fleet['status'].isna()]


# In[14]:


df_fleet[df_fleet['type'].isna()]


# In[15]:


df_fleet[df_fleet['serial number'].isna()]


# In[16]:


df_fleet[df_fleet['operator'].isna()]


# In[17]:


df_fleet[df_fleet['master series'].isna()]


# In[18]:


df_fleet[df_fleet['status'].isna()]


# In[19]:


df_fleet[df_fleet['manufacturer'].isna()]


# In[20]:


df_fleet[df_fleet['in service date'].isna()]


# In[21]:


df_fleet[df_fleet['registration'].isna()]


# In[22]:


df_fleet[df_fleet['year'].isna()]


# In[23]:


# Columns containing null values are:
# in service date and registration


# In[24]:


# Checking the the count of different values a column has


# In[25]:


df_fleet['status'].str.lower().value_counts(dropna=False)


# In[26]:


# Filtering out rows in status column


# In[27]:


df_fleet = df_fleet[~df_fleet['status'].str.lower().isin(['loi to order', 'on order'])]


# In[28]:


df_fleet['status'].str.lower().value_counts(dropna=False)


# In[29]:


df_fleet['serial number'].str.lower().value_counts(dropna=False) #this shows that the serial numbers are not unique.


# In[30]:


df_fleet['operator'].str.lower().value_counts(dropna=False)


# In[31]:


df_fleet['type'].str.lower().value_counts(dropna=False)


# In[32]:


df_fleet['manufacturer'].str.lower().value_counts(dropna=False)


# In[33]:


df_fleet['type'].str.lower().value_counts(dropna=False)


# In[34]:


df_fleet['master series'].str.lower().value_counts(dropna=False)


# In[35]:


df_fleet['in service date'].str.lower().value_counts(dropna=False)


# In[36]:


# checking the datatypes of the columns
datatypes = df_fleet.dtypes


# In[37]:


datatypes


# In[38]:


# Problem Statement 1: 
# Exact count of aircrafts are “currently operating” for a particular operator in a particular year at a master series level.


# In[39]:


# Check how many years each aircraft (serial number) appears in


# In[40]:


# serial_years = df_fleet.groupby('serial number')['year'].nunique().sort_values(ascending=False)
# serial_years


# In[48]:


# # Group by year and get all serial numbers
# year_serials = {
#     year: set(df_fleet[df_fleet['year'] == year]['serial number'].dropna())
#     for year in sorted(df_fleet['year'].unique())
# }


# In[49]:


# #Compare years (new, exits, retired)

# changes = []

# years_sorted = sorted(year_serials.keys())

# for i in range(len(years_sorted)):
#     year = years_sorted[i]
#     current = year_serials[year]

#     # First year: only "new" and "retired"
#     if i == 0:
#         retired_serials = set(
#             df_fleet[(df_fleet['year'] == year) & (df_fleet['status'] == 'retired')]['serial number']
#         )
#         changes.append({
#             'year': year,
#             'new_entries': len(current),
#             'exits': 0,
#             'retired': len(retired_serials)
#         })
#     else:
#         prev = year_serials[years_sorted[i - 1]]
#         new_entries = current - prev
#         exits = prev - current

#         retired_serials = set(
#             df_fleet[(df_fleet['year'] == year) & (df_fleet['status'] == 'retired')]['serial number']
#         )

#         changes.append({
#             'year': year,
#             'new_entries': len(new_entries),
#             'exits': len(exits),
#             'retired': len(retired_serials)
#         })

# # Create DataFrame
# df_changes = pd.DataFrame(changes)


# In[50]:


# df_changes


# In[51]:


# building a new column for identifying a primary key by concatenating serial number, master series and manufacturer.

df_fleet['unique Identifier'] = df_fleet['serial number'].astype(str) + '|' + df_fleet['master series'].astype(str) + '|' + df_fleet['manufacturer'].astype(str)


# In[52]:


df_fleet = df_fleet.copy()

df_fleet['unique Identifier'] = (
    df_fleet['serial number'].astype(str) + '|' +
    df_fleet['master series'].astype(str) + '|' +
    df_fleet['manufacturer'].astype(str)
)


# In[53]:


df_fleet


# In[54]:


# status
# in service       302311
# cancelled        160933
# on order         112159
# retired           72242
# on option         60659
# written off       25491
# storage           15912
# loi to option     12594
# loi to order      11863
# unknown              74
# Name: count, dtype: int64


# In[58]:


# Building a Flag column for accurately identifying the currently operating aircrafts.

df_fleet['in_service_loi_flag'] = df_fleet.apply(
    lambda row: (
        'Y' if pd.notnull(row['in service date']) and row['status'].strip().lower() == 'loi to option'
        else 'N' if pd.isnull(row['in service date']) and row['status'].strip().lower() == 'loi to option'
        else 'Z'
    ),
    axis=1
)


# In[102]:


print(df_fleet['in_service_loi_flag'].value_counts())


# In[61]:


df_fleet


# In[63]:


# Filter only "in service" records
filtered_df = df_fleet[
    (df_fleet['in_service_loi_flag'] == 'Y') |
    (
        (df_fleet['in_service_loi_flag'] == 'Z') &
        (df_fleet['status'].str.lower().isin(['in service', 'storage']))
    )
]

# Final result
result = (
    filtered_df
    .groupby(['year', 'operator', 'master series'])['unique Identifier']
    .nunique()
    .reset_index(name='currently operating')
)


# In[64]:


result


# In[65]:


result_OperatorWise = (
    filtered_df
    .groupby(['year', 'operator'])['unique Identifier']
    .nunique()
    .reset_index(name='currently operating')
)


# In[66]:


result_OperatorWise


# In[67]:


# Year-wise count
result_YearWise = (
    filtered_df
    .groupby(['year'])['unique Identifier']
    .nunique()
    .reset_index(name='currently operating')
)

# Grand total row
grand_total = pd.DataFrame({
    'year': ['Grand Total'],
    'currently operating': [result_YearWise['currently operating'].sum()]
})

# Append grand total to the result
result_YearWise = pd.concat([result_YearWise, grand_total], ignore_index=True)


# In[68]:


result_YearWise


# In[69]:


# Bar Plot: Trend of Operating Aircraft by Year


# In[164]:


import matplotlib.pyplot as plt
import seaborn as sns

# Aggregate by year for bar plot
yearly_summary = result.groupby('year')['currently operating'].sum().reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(data=yearly_summary, x='year', y='currently operating', palette='Blues_d')
plt.title('Total "Currently Operating Aircrafts" YoY')
plt.xlabel('Year')
plt.ylabel('Count of Aircrafts')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[71]:


# Top 25 Operators


# In[72]:


top_operators = (
    result.groupby('operator')['currently operating']
    .sum()
    .sort_values(ascending=False)
    .head(25)
    .reset_index()
)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_operators, y='operator', x='currently operating', palette='viridis')
plt.title('Top 25 Operators by Total Operating Aircrafts')
plt.xlabel('Total Operating Aircrafts')
plt.ylabel('Operator')
plt.tight_layout()
plt.show()


# In[73]:


# Line Plot: Master Series Evolution


# In[74]:


top_series = result['master series'].value_counts().head(10).index
series_trend = result[result['master series'].isin(top_series)]

plt.figure(figsize=(14, 6))
sns.lineplot(
    data=series_trend,
    x='year', y='currently operating',
    hue='master series', marker='o'
)
plt.title('Top 10 Aircraft Master Series Operating Trend')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend(title='Master Series')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[75]:


# Problem Statement 3:
# Exact count of new aircrafts that have been added in that particular year for an airline at a master series level.


# In[76]:


# Sort the data to prepare for first appearance detection
df_valid = df_fleet.sort_values(by=['serial number', 'operator', 'year'])

# Identify first appearance of each serial number for each operator
df_first_appearance = (
    df_valid
    .drop_duplicates(subset=['serial number', 'operator'], keep='first')
)

# Group by year, operator, and master series to count unique serial numbers.
new_aircraft_counts = (
    df_first_appearance
    .groupby(['year', 'operator', 'master series'])
    .agg(new_aircrafts_added=('serial number', 'nunique'))
    .reset_index()
)


# In[77]:


new_aircraft_counts


# In[78]:


# Group by year and operator, 
# then sum new aircrafts added
new_aircrafts_yearly_sum = (
    new_aircraft_counts
    .groupby(['year', 'operator'])['new_aircrafts_added']
    .sum()
    .reset_index()
    .sort_values(['year', 'new_aircrafts_added'], ascending=[True, False])
)


# In[81]:


new_aircrafts_yearly_sum.head(10)


# In[82]:


# Line Chart: Total New Aircrafts Added Over Time


# In[83]:


total_new_aircrafts = (
    new_aircrafts_yearly_sum
    .groupby('year')['new_aircrafts_added']
    .sum()
    .reset_index()
)

plt.figure(figsize=(12, 6))
sns.lineplot(data=total_new_aircrafts, x='year', y='new_aircrafts_added', marker='o')

plt.title('Total New Aircrafts Added Across All Operators Over Time')
plt.xlabel('Year')
plt.ylabel('Total New Aircrafts Added')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[84]:


# Sum of all new aircrafts added per year
total_new_aircrafts_per_year = (
    new_aircraft_counts
    .groupby('year')['new_aircrafts_added']
    .sum()
    .reset_index()
    .rename(columns={'new_aircrafts_added': 'total_new_aircrafts'})
)


# In[85]:


print(total_new_aircrafts_per_year.head(50))


# In[86]:


# Stacked Area Chart: Year-wise Composition by operator


# In[97]:


# Pivot to reshape for stacked area chart
pivot_df = new_aircraft_counts.pivot_table(
    index='year',
    columns='operator',
    values='new_aircrafts_added',
    aggfunc='sum',
    fill_value=0
)

# Plot
pivot_df.plot.area(figsize=(20, 10), colormap='tab20')
plt.title('New Aircrafts Added Per Year by Operator')
plt.xlabel('Year')
plt.ylabel('Count of New Aircrafts')
plt.legend(title='Master Series', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[ ]:


# Top 10 Operator-Master Series Pairs by Total New Aircrafts


# In[100]:


top_pairs = (
    new_aircraft_counts
    .groupby(['operator', 'master series'])['new_aircrafts_added']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=top_pairs,
    x='new_aircrafts_added',
    y=top_pairs['operator'] + ' - ' + top_pairs['master series'],
    palette='viridis'
)
plt.title('Top 10 Operator - Master Series by Total New Aircrafts')
plt.xlabel('New Aircrafts Added')
plt.ylabel('Operator - Master Series')
plt.tight_layout()
plt.show()


# In[101]:


# Problem Statement 4:
# Average age of each Master Series within an airline operator for each year


# In[103]:


# defining a reusable function


# In[104]:


def is_in_service(df):
    return (
        (df['in_service_loi_flag'] == 'Y') |
        (
            (df['in_service_loi_flag'] == 'Z') &
            (df['status'].str.lower().isin(['in service', 'storage']))
        )
    )


# In[108]:


# converting 'in service date' in datetime format
df_fleet['in service date'] = pd.to_datetime(df_fleet['in service date'], errors='coerce')

# Earliest in-service date for each serial number
serial_to_first_service = (
    df_fleet.dropna(subset=['in service date'])
    .groupby('serial number')['in service date']
    .min()
    .reset_index()
    .rename(columns={'in service date': 'first_in_service_date'})
)

# Merging earliest in-service date into main dataset
df_with_first_service = df_fleet.merge(serial_to_first_service, on='serial number', how='left')

# Filtering only 'in service' records using the reusable function
df_active = df_with_first_service[is_in_service(df_with_first_service)].copy()

# To calculate the average age = year - first_in_service_date.year
df_active['age'] = df_active['year'] - df_active['first_in_service_date'].dt.year

# Group by year, operator, and master series, and final average age
avg_age_summary = (
    df_active
    .groupby(['year', 'operator', 'master series'])['age']
    .mean()
    .reset_index()
    .rename(columns={'age': 'avg_age'})
    .sort_values(['year', 'operator'])
)


# In[109]:


avg_age_summary


# In[110]:


# Heatmap: Operator vs Year by Average Age


# In[116]:


# Pivot to form operator vs year matrix
heatmap_data = avg_age_summary.pivot_table(
    index='operator', columns='year', values='avg_age', aggfunc='mean'
)

# Filter top operators (optional for readability)
top_heatmap = heatmap_data.loc[heatmap_data.mean(axis=1).nlargest(10).index]

# Plot
plt.figure(figsize=(20, 8))
sns.heatmap(top_heatmap, annot=True, fmt=".1f", cmap="coolwarm", linewidths=0.5)
plt.title('Average Aircraft Age: Operator vs Year')
plt.xlabel('Year')
plt.ylabel('Operator')
plt.tight_layout()
plt.show()


# In[122]:


avg_age_summary.tail()


# In[123]:


# Top 10 Operators with Oldest Fleets (Latest Year)


# In[124]:


latest_year = avg_age_summary['year'].max()
top_old_operators = (
    avg_age_summary[avg_age_summary['year'] == latest_year]
    .groupby('operator')['avg_age']
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12,6))
sns.barplot(x=top_old_operators.values, y=top_old_operators.index, palette='magma')
plt.title(f'Top 10 Operators by Average Fleet Age in {latest_year}')
plt.xlabel('Average Age')
plt.ylabel('Operator')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[125]:


# Top Manufacturers with Highest Average Aircraft Age by Series


# In[135]:


df_active['manufacturer'] = df_active['manufacturer'].fillna('Unknown')
manu_series_avg_age = (
    df_active
    .groupby(['manufacturer', 'master series'])['age']
    .mean()
    .reset_index()
    .sort_values(by='age', ascending=False)
)

top_manufacturer_series = manu_series_avg_age.head(10)

plt.figure(figsize=(16, 6))
sns.barplot(
    data=top_manufacturer_series,
    x='age', y='master series', hue='manufacturer', dodge=False
)
plt.title('Top Manufacturer-Series Combos with Highest Avg Age')
plt.xlabel('Average Age')
plt.ylabel('Master Series')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[132]:


# Boxplot: Distribution of Aircraft Age by Operator (Latest Year)


# In[136]:


plt.figure(figsize=(16, 6))
sns.boxplot(
    data=df_active[df_active['year'] == latest_year],
    x='operator', y='age',
    order=df_active[df_active['year'] == latest_year]['operator'].value_counts().head(10).index
)
plt.title(f'Aircraft Age Distribution by Operator in {latest_year}')
plt.xlabel('Operator')
plt.ylabel('Age')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[137]:


# Problem Statement 2:
# Exact count of aircrafts that have exited and retired in a particular year for an airline at a master series level.


# In[144]:


# Identifying 'in service' birds
flyable_df = df_fleet[is_in_service(df_fleet)][
    ['key', 'year', 'operator', 'master series']
].copy()

# Renaming columns
flyable_df = flyable_df.rename(columns={
    'year': 'flyable_year',
    'operator': 'flyable_operator',
    'master series': 'flyable_master_series'
})


# In[145]:


# Merging to find all future flyable records for same aircraft key
merged = df_fleet.merge(
    flyable_df,
    on='key',
    how='left',
    suffixes=('', '_flyable')
)

# Filter only future flyable records (flyable_year > current year)
merged = merged[merged['flyable_year'] > merged['year']]


# In[153]:


# For each original row, check:
# 1) Is there any future flyable record with same operator & master series?
merged['FutureSameOperator'] = (
    (merged['operator'] == merged['flyable_operator']) &
    (merged['master series'] == merged['flyable_master_series'])
)

# 2) Is there any future flyable record with different operator but same master series?
merged['FutureOtherOperator'] = (
    (merged['operator'] != merged['flyable_operator']) &
    (merged['master series'] == merged['flyable_master_series'])
)


# In[154]:


# Now aggregate these booleans per original row index:
agg = merged.groupby(merged.index).agg({
    'FutureSameOperator': 'max',  # True if any True
    'FutureOtherOperator': 'max'
})

# Join back these flags to original df
df_fleet = df_fleet.join(agg)

# Fill missing (no future flyable record) with False
df_fleet['FutureSameOperator'] = df_fleet['FutureSameOperator'].fillna(False)
df_fleet['FutureOtherOperator'] = df_fleet['FutureOtherOperator'].fillna(False)

# Classify exit_type
def classify(row):
    status = row['status_lower']
    if status not in ['retired', 'written off']:
        return None
    
    if row['FutureSameOperator']:
        # Still flying with same operator in future, so not retired/exited
        return None
    elif row['FutureOtherOperator']:
        # Transferred to different operator, so exited
        return 'exited'
    else:
        # No future flyable anywhere, retired
        return 'retired'

df_fleet['exit_type'] = df_fleet.apply(classify, axis=1)


# In[155]:


# Finally group and summarize as before
summary = (
    df_fleet[df_fleet['exit_type'].notnull()]
    .groupby(['year', 'operator', 'master series', 'exit_type'])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

if 'retired' not in summary.columns:
    summary['retired'] = 0
if 'exited' not in summary.columns:
    summary['exited'] = 0

summary = summary.rename(columns={'retired': 'retired_count', 'exited': 'exited_count'})
summary = summary[['year', 'operator', 'master series', 'retired_count', 'exited_count']]


# In[156]:


# summary


# In[157]:


# Ensure numeric type
summary['retired_count'] = summary['retired_count'].astype(int)
summary['exited_count'] = summary['exited_count'].astype(int)

# Grand totals
total_retired = summary['retired_count'].sum()
total_exited = summary['exited_count'].sum()

print(f"Total Retired Aircraft: {total_retired}")
print(f"Total Exited Aircraft: {total_exited}")


# In[158]:


# Aircraft Exit/Retirement Trend Over Time


# In[159]:


# Aggregate counts per year and exit type
trend = (
    df_fleet[df_fleet['exit_type'].notna()]
    .groupby(['year', 'exit_type'])['serial number']
    .nunique()
    .reset_index()
)

# Plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=trend, x='year', y='serial number', hue='exit_type', marker='o')
plt.title('Trend of Exited vs Retired Aircraft per Year')
plt.ylabel('Number of Aircraft')
plt.xlabel('Year')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[160]:


# Top Operators with Most Retirements/Exits


# In[161]:


# Aggregate total exits/retirements by operator
top_ops = (
    df_fleet[df_fleet['exit_type'].notna()]
    .groupby(['operator', 'exit_type'])['serial number']
    .nunique()
    .reset_index()
)

# Top 10 operators overall
top_10_ops = (
    top_ops.groupby('operator')['serial number'].sum()
    .nlargest(10).index.tolist()
)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=top_ops[top_ops['operator'].isin(top_10_ops)],
    y='operator', x='serial number', hue='exit_type'
)
plt.title('Top 10 Operators by Retired/Exited Aircraft')
plt.xlabel('Number of Aircraft')
plt.ylabel('Operator')
plt.tight_layout()
plt.show()


# In[162]:


# Pie Chart: Total Distribution


# In[163]:


exit_pie = (
    df_fleet['exit_type']
    .value_counts()
    .rename_axis('exit_type')
    .reset_index(name='count')
)

plt.figure(figsize=(6, 6))
plt.pie(
    exit_pie['count'],
    labels=exit_pie['exit_type'],
    autopct='%1.1f%%',
    startangle=140,
    colors=['#66c2a5', '#fc8d62']
)
plt.title('Overall Exit vs Retirement Distribution')
plt.show()


# In[174]:


# Unfolding Logics Problems Wise


# In[175]:


# Problem 1:
# Building a unique flag column using Status and In Service Date to help identifying the correct "Currently Operating" count YoY.

# Problem 2:
# Retired:
# If status == 'retired' or status == 'written off' and the aircraft never flies again.
# Exited:
# If aircraft with status == 'retired' or 'written off' later appears again (same serial number, master series, and manufacturer) 
# but with a different operator.

# Problem 3:
# Must detect when a serial number first appears with a new operator.
# Can’t use in service date directly due to reuse across operators.

# Problem 4:
# Detect the serial number having earliest service date.
# Compute age = year - earliest in-service year.


# In[ ]:




