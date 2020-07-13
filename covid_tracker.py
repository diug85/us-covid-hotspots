import requests
import csv
import smtplib
import os
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PyPDF2 import PdfFileMerger
from os.path import basename
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from sklearn.mixture import GaussianMixture

FROM = 'brown.institute.heroku@gmail.com' # os.environ.get('FROM')
PASSW = 'trumptown1' # os.environ.get('PASSW')
TO = ['drk2134@columbia.edu ',
        'du2160@columbia.edu',
        'gmg2172@columbia.edu',
        'kas2317@columbia.edu',
        'mh3287@columbia.edu',
        'juan.saldarriaga@columbia.edu',
        'alexander.c@columbia.edu']
SUBJECT = '[AUTOMATED] Covid tracker {}'

def get_csv_file():
    '''
    Download the latest version of the covid tracker on the county level from
    https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv

    Then formats dates and numerical columns
    '''
    url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'
    content = requests.get(url).content.decode('utf-8')
    reader = csv.reader(content.splitlines())
    columns = reader.__next__() #column names in 1st row
    df = pd.DataFrame(reader, columns=columns)

    # format columns
    df['date'] = pd.to_datetime(df.date, infer_datetime_format=True)
    df['cases'] = pd.to_numeric(df.cases)
    df['deaths'] = pd.to_numeric(df.deaths)

    # sort data and fillna
    df = df.sort_values(['state','county','date'])
    df.fillna(0, inplace=True) # fill missins 'fips' with 0

    return df

def transform_data(grouped, date='2020-05-15'):
    '''
    Calculates daiyly new casesa and moving averages for new cases

    INPUT
    - grouped: pd.DataFrame().groupby dataframe, grouped by 'state' and 'county'
    - date: a reference date (a base date) to rescale moving averages

    OUTPUT
    An UNGROUPED dataframe adding 3 columns. For each group it calculates:
    - cases_delta: new daily cases for each date
    - cases_mov_avg: 7-day moving average for new cases
    - scaled_mov_avg: rescales 'cases_mov_avg' dividing it by its value on the
        date given as a parameter

    '''
    delta = grouped.cases.diff()
    ma = delta.rolling(7).mean()
    scaled = scale(pd.DataFrame({'date': grouped.date,  'cases_mov_avg': ma}),  date=date).array

    df = pd.DataFrame({'state': grouped.state,
                       'county': grouped.county,
                       'date': grouped.date,
                       'cases': grouped.cases,
                       'cases_delta': delta,
                       'cases_mov_avg': ma,
                       'scaled_mov_avg': scaled})
    return df


def find_closest_date(grouped, date):
    date = pd.to_datetime(date)
    diff = pd.to_numeric(grouped.date - date)
    idx = (diff>=0) & (grouped.cases_mov_avg>0)
    return grouped.date[idx].min()


def scale(grouped, date='2020-05-15'):
    date = find_closest_date(grouped, date)
    idx = grouped.date==date

    if sum(idx)==0 or pd.isnull(date):
        return pd.Series(len(grouped)*[None], name='scaled_mov_avg')

    scaled = grouped.cases_mov_avg / grouped[idx].cases_mov_avg.array
    scaled.rename('scaled_mov_avg', inplace=True)
    return scaled


def cluster_county_manual(cases):
    if cases<=500:
        return 0
    elif cases<=4000:
        return 1
    else:
        return 2

def cluster_county_gaussian(latest_cases, n_clusters=4, random_state=0):
    gaussian = GaussianMixture(n_clusters, random_state=random_state)
    X = latest_cases.cases.to_numpy().reshape(-1, 1)
    gaussian.fit(X)
    clusters = gaussian.predict(X)

    return pd.DataFrame(clusters+1, index=latest_cases.index, columns=['cluster'])


def cases_summary(trends):
    grouped_trends = trends.groupby(['state','county'], sort=False)
    latest = grouped_trends.last()
    return latest


def top_counties(trends, by, top=5):
    largest = trends[by].nlargest(top)
    return list(largest.index)


def filter_by_cluster_date(trends, cluster_no, date):
    idx = (trends.date>=base_date) & (trends.cluster==cluster_no)
    return trends[idx]


def plot_county(df, ax, in_color, state=None, county=None):
    if in_color:
        lbl = county + ', ' + state
        ax.plot(df.iloc[:,0], df.iloc[:,1], label=lbl)
        ax.legend()
    else:
        ax.plot(df.iloc[:,0], df.iloc[:,1], color='gray', lw=0.25)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))


def plot_cluster(df, colored_list, y_column, title, file_name=None, n=250):
    counties = df[['state','county']].drop_duplicates()
    title = '{title} ({cnt} counties)'.format(title=title, cnt=len(counties))

    fig, ax = plt.subplots(figsize=(16,8))
    ax.set_title(title)
    ax.set_ylabel('7-day Moving Average')

    i = 0
    n -= len(colored_list)
    for row in counties.iterrows():
        state = row[1][0]
        county = row[1][1]

        if (state, county) in colored_list:
            continue

        idx = (df.state==state) & (df.county==county)
        plot_county(df=df[idx][['date',y_column]],
                    ax=ax,
                    in_color=False)
        i += 1
        if i==n:
            break

    y_max = 1
    for state, county in colored_list:
        idx = (df.state==state) & (df.county==county)
        y_max = max(y_max, df[idx][y_column].max())
        plot_county(df=df[idx][['date',y_column]],
                    ax=ax,
                    in_color=True,
                    state=state,
                    county=county)

    ax.set_ylim((0, y_max*1.025))

    if file_name:
        fig.savefig(file_name)

def df_deltas(trends):
    column_names = ['date','total_cases', 'new_cases','new_cases_ma','scaled_new_cases_ma','cluster']
    df_dates = trends.date.drop_duplicates().nlargest(2) # get last 2 dates
    # latest date
    idx = trends.date==df_dates.iloc[0]
    B = trends[idx].reset_index(drop=True).set_index(['state','county'])
    B.columns = column_names
    # day before
    idx = trends.date==df_dates.iloc[1]
    A = trends[idx].reset_index(drop=True).set_index(['state','county'])
    A.columns = column_names
    # deltas
    deltas = B-A
    deltas = deltas.sort_values(['cluster','new_cases_ma'], ascending=[True, False])
    # re-order A and B
    A = A.reindex(deltas.index)
    B = B.reindex(deltas.index)

    output_dct = {str(df_dates.iloc[1])[:10]: A,
                  str(df_dates.iloc[0])[:10]: B,
                  'deltas': B-A}
    return output_dct

def export_xlsx(df_dict, file_name):
    with pd.ExcelWriter(file_name) as writer:
        for sheet, df in df_dict.items():
            df.to_excel(writer, sheet_name=sheet)


def merge_pdfs(pdf_list, file_name):
    merger = PdfFileMerger()
    for f in pdf_list:
        merger.append(f)
    merger.write(file_name)
    merger.close()


def send_mail(to, subject, body, attachment=[]):
    #Setup the MIME
    message = MIMEMultipart()
    message['From'] = FROM
    message['To'] = ', '.join(to)
    message['Subject'] = subject

    #The body and the attachments for the mail
    message.attach(MIMEText(body, 'html'))

    for file in attachment:
        with open(file, 'rb') as f:
            part = MIMEApplication(f.read(), Name=basename(file))
            part['Content-Disposition'] = 'attachment; filename="{}"'.format(basename(file))
            message.attach(part)

    #Create SMTP session for sending the mail
    session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
    session.starttls() #enable security
    session.login(FROM, PASSW) #login with mail_id and password
    text = message.as_string()
    session.sendmail(FROM, to, text)
    session.quit()
    print('Mail Sent')


if __name__=="__main__":
    base_date = '2020-05-15'
    n_clusters = 4
    today = str(dt.date.today())

    data = get_csv_file()
    grouped = data.groupby(['state','county'], sort=False)
    trends = grouped.apply(lambda x: transform_data(x, base_date))
    latest_cases = cases_summary(trends)
    clusters = cluster_county_gaussian(latest_cases, n_clusters=n_clusters, random_state=0)
    latest_cases['cluster'] = clusters.cluster

    trends = trends.merge(clusters, how='left', left_on=['state','county'], right_index=True)
    top_by_new_cases = latest_cases.groupby('cluster').apply(top_counties, 'cases_mov_avg', 5)
    top_by_increase_rate = latest_cases.groupby('cluster').apply(top_counties, 'scaled_mov_avg', 5)

    for i in range(1, n_clusters+1):
        cluster_ = filter_by_cluster_date(trends, i, base_date)

        plot_title = 'NEW CASES\nCluster ' + str(i) + '\n'
        file_name = 'new_cases_cluster_' + str(i) + '.pdf'
        plot_cluster(cluster_, top_by_new_cases[i], 'cases_mov_avg', plot_title, file_name)

        plot_title = 'SCALED NEW CASES \nBase: ' + base_date +'\nCluster ' + str(i) + '\n'
        file_name = 'scaled_new_cases_cluster_' + str(i) + '.pdf'
        plot_cluster(cluster_, top_by_increase_rate[i], 'scaled_mov_avg', plot_title, file_name)

    deltas = df_deltas(trends)
    file_name = 'covid_track_' + today
    export_xlsx(deltas, file_name + '.xlsx')

    pdf_list = ['new_cases_cluster_{}.pdf'.format(i) for i in range(1, n_clusters+1)] + ['scaled_new_cases_cluster_{}.pdf'.format(i) for i in range(1, n_clusters+1)]

    merge_pdfs(pdf_list, file_name + '.pdf')
    #
    # send_mail( TO, SUBJECT.format(today), '', attachment=[file_name + '.xlsx', file_name + '.pdf'])
    send_mail(['ivan.u@columbia.edu'], SUBJECT.format(today), '', attachment=[file_name + '.xlsx', file_name + '.pdf'])
