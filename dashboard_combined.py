import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import plotly.express as px
from pathlib import Path
from io import StringIO
import io
import plotly.graph_objects as go

# Page Configure
st.set_page_config(page_title="Employment Dashboard", page_icon="📊", layout="wide")

# --- Sidebar selector ---
st.sidebar.title("🔍 Select Analysis Focus")
choice = st.sidebar.radio(
    "Choose analysis dimension:",
    ["👩 Gender", "👵 Age"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("Switch between dashboards to explore employment patterns by gender or age group.")

# Data Preprocessing Function
def preprocess_data(df):
    df.iloc[:,1:]=df.iloc[:,1:].apply(pd.to_numeric,errors='coerce')
    df.columns=[c.strip() for c in df.columns]
    df['Data Series']=df['Data Series'].str.strip()

# Read data as dataframes
#  Employed Male Residents Aged 15 Years And Over By Industry And Age Group
df_m_ind=pd.read_csv("M182141-table.csv",skiprows=9,skipfooter=113)
# Employed Female Residents Aged 15 Years And Over By Industry And Age Group
df_f_ind=pd.read_csv("M182151-table.csv",skiprows=9,skipfooter=113)
# Number Of Full-Time Employed Residents By Sex
df_ft=pd.read_csv("M920141-table.csv",skiprows=10,skipfooter=17)
# Number Of Part-Time Employed Residents By Sex
df_pt=pd.read_csv("M920151-table.csv",skiprows=10,skipfooter=17)

# Data preparation
# male
preprocess_data(df_m_ind)
# female
preprocess_data(df_f_ind)
# full-time
preprocess_data(df_ft)
# part_time
preprocess_data(df_pt)

# employed male residents by industries
df_1=df_m_ind.iloc[[1,2,3,16],:]
# transfer to long table
df_1_long = df_1.melt(id_vars="Data Series", 
                    var_name="Year", 
                    value_name="Employment")
df_1_long['Gender']='male'


# employed female residents by industries
df_2=df_f_ind.iloc[[1,2,3,16],:]
# transfer to long table
df_2_long = df_2.melt(id_vars="Data Series", 
                    var_name="Year", 
                    value_name="Employment")
df_2_long['Gender']='female'

# concat female and male data
df_concat=pd.concat([df_1_long,df_2_long],ignore_index=True)
df_concat.rename({'Data Series':'Industry'},axis=1,inplace=True)

# Funtion to transfer wide table into long table (applicable to full time/part time data)
def transfer_long_table(df):
    df_long=df.melt(id_vars='Data Series',
                    var_name='Year',
                    value_name='Employment')
    df_total=df_long[df_long['Data Series']=='Total'].iloc[:,1:]
    df_total.rename({'Employment':'Total Employment'},axis=1,inplace=True)
    df_female=df_long[df_long['Data Series']=='Female'].iloc[:,1:]
    df_plot=pd.merge(df_female,df_total,on='Year')
    # compute female prop
    df_plot['Prop.']=df_plot['Employment']/df_plot['Total Employment']*100
    return df_plot


# User Interface
if choice == "👩 Gender":
    # Title
    st.title("Employment Insights by Gender 👨‍💼👩‍💼")
    # Chart 1
    col1, col2 = st.columns((2))

    with col1:
        st.subheader("🧑‍💻 Occupational Gender Distribution")
        female_df = pd.read_csv("M182191-table.csv", skiprows=10, skipfooter=154, engine='python')
        male_df = pd.read_csv("M182181-table.csv", skiprows=10, skipfooter=154, engine='python')

        female_df.columns = [c.strip() for c in female_df.columns]
        male_df.columns = [c.strip() for c in male_df.columns]

        years = [str(y) for y in range(2000, 2025)]
        latest_year = st.selectbox(
        "📅 Select year:",
        years[::-1],
        index=0
        )

        occupations = female_df.iloc[1:10, 0].reset_index(drop=True)
        female_data = female_df.iloc[1:10, female_df.columns.get_loc(latest_year)].astype(float).reset_index(drop=True)
        male_data = male_df.iloc[1:10, male_df.columns.get_loc(latest_year)].astype(float).reset_index(drop=True)

        plot_df = pd.DataFrame({
            'Occupation': occupations,
            'Female': female_data,
            'Male': male_data
        })
        plot_df['Total'] = plot_df['Female'] + plot_df['Male']
        plot_df = plot_df.sort_values('Total', ascending=False)
        plot_df['Female_Ratio'] = plot_df['Female'] / plot_df['Total']
        plot_df['Male_Ratio'] = plot_df['Male'] / plot_df['Total']

        fig, ax = plt.subplots(figsize=(14, 8))
        bars_female = ax.bar(plot_df['Occupation'], plot_df['Female'], label='Female', color='pink')
        bars_male = ax.bar(plot_df['Occupation'], plot_df['Male'], bottom=plot_df['Female'], label='Male', color='lightblue')

        for i, (female_val, male_val, total_val) in enumerate(zip(plot_df['Female'], plot_df['Male'], plot_df['Total'])):
            if total_val > 0:
                female_percent = (female_val / total_val) * 100
                male_percent = (male_val / total_val) * 100
                ax.text(i, female_val/2, f'{female_percent:.1f}%', ha='center', va='center', fontweight='bold', fontsize=9)
                ax.text(i, female_val + male_val/2, f'{male_percent:.1f}%', ha='center', va='center', fontweight='bold', fontsize=9)

        ax.set_title(f'Employment by Occupation and Gender ({latest_year})', fontsize=16, pad=20)
        ax.set_xlabel('Occupation', fontsize=12)
        ax.set_ylabel('Number of Employees', fontsize=12)
        ax.set_xticklabels(plot_df['Occupation'], rotation=45, ha='right')
        ax.legend()

        st.pyplot(fig)

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📋 Data Table")
        st.dataframe(plot_df[['Occupation', 'Female', 'Male', 'Total']].round(3),hide_index=True)

    # Chart 2
    st.markdown('---')
    st.subheader("📊 Female Ratio Over Time for a Selected Occupation")

    year_columns = [col for col in female_df.columns if col.strip().isdigit() and len(col.strip()) == 4]
    year_columns.sort()

    occupation_options = [c.strip() for c in female_df.iloc[1:10, 0].tolist()]
    selected_occ = st.selectbox("🧑‍🏭 Select occupation:", occupation_options)

    female_ratios = []
    years_list = []

    for year in year_columns:
        female_value = float(female_df.loc[female_df.iloc[:, 0].str.strip() == selected_occ, year].values[0])
        male_value = float(male_df.loc[male_df.iloc[:, 0].str.strip() == selected_occ, year].values[0])
        total = female_value + male_value
        ratio = female_value / total if total > 0 else 0
        female_ratios.append(ratio)
        years_list.append(int(year))

    occ_trend_df = pd.DataFrame({
        "Year": years_list,
        "Female Ratio": female_ratios
    })

    fig_occ = px.line(
        occ_trend_df,
        x="Year",
        y="Female Ratio",
        markers=True,
        color_discrete_sequence=["deeppink"],
        template="plotly_white"
    )

    fig_occ.update_traces(
        hovertemplate="<b>%{x}</b><br>Female Ratio: %{y:.1%}",
        line=dict(width=3),
        marker=dict(size=8)
    )
    fig_occ.update_layout(
        yaxis_tickformat=".0%",
        yaxis_title="Female Ratio",
        xaxis_title="Year",
        title={
        'text': f"Female Ratio Trend for '{selected_occ}'",
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
        },
        height=500,
        width=1000

    )
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.plotly_chart(fig_occ, use_container_width=False)


    # Chart 3
    st.subheader('👩‍🧑 Gender Distribution in Technical and Non-technical Occupations')
    latest_year = st.selectbox("📅 Select year:", [str(y) for y in range(2000, 2025)], index=24)

    occupations = female_df.iloc[1:10, 0].reset_index(drop=True)
    occupations2 = [c.strip() for c in occupations]
    female_data = female_df.iloc[1:10, female_df.columns.get_loc(latest_year)].astype(float).reset_index(drop=True)
    male_data = male_df.iloc[1:10, male_df.columns.get_loc(latest_year)].astype(float).reset_index(drop=True)

    plot_df = pd.DataFrame({
        'Occupation': occupations2,
        'Female': female_data,
        'Male': male_data
    })

    technical_occupations = [
        'Managers & Administrators (Including Working Proprietors)',
        'Professionals',
        'Associate Professionals & Technicians',
        'Craftsmen & Related Trade Workers',
        'Plant & Machine Operators & Assemblers',
        'Others'
    ]

    non_technical_occupations = [
        'Clerical Support Workers',
        'Service & Sales Workers',
        'Cleaners, Labourers & Related Workers'
    ]

    plot_df['Category'] = plot_df['Occupation'].apply(
        lambda x: 'Technical' if x in technical_occupations else 'Non-Technical'
    )

    category_stats = plot_df.groupby('Category').agg({
        'Female': 'sum',
        'Male': 'sum'
    }).reset_index()


    category_stats['Total'] = category_stats['Female'] + category_stats['Male']
    category_stats['Female_Ratio'] = category_stats['Female'] / category_stats['Total']
    category_stats['Male_Ratio'] = category_stats['Male'] / category_stats['Total']

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    categories = category_stats['Category']
    female_counts = category_stats['Female']
    male_counts = category_stats['Male']

    ax1.bar(categories, female_counts, label='Female', color='pink', alpha=0.8)
    ax1.bar(categories, male_counts, bottom=female_counts, label='Male', color='lightblue', alpha=0.8)
    ax1.set_title('Employment by Technical vs Non-Technical Categories')
    ax1.set_ylabel('Number of Employees')
    ax1.legend()


    tech_female_ratio = category_stats[category_stats['Category'] == 'Technical']['Female_Ratio'].values[0]
    tech_male_ratio = category_stats[category_stats['Category'] == 'Technical']['Male_Ratio'].values[0]
    nontech_female_ratio = category_stats[category_stats['Category'] == 'Non-Technical']['Female_Ratio'].values[0]
    nontech_male_ratio = category_stats[category_stats['Category'] == 'Non-Technical']['Male_Ratio'].values[0]


    ax2.pie([tech_female_ratio, tech_male_ratio],
            labels=['Female', 'Male'],
            colors=['pink', 'lightblue'],
            autopct='%1.1f%%', startangle=90)
    ax2.set_title('Technical Occupations: Gender Ratio')

    ax3.pie([nontech_female_ratio, nontech_male_ratio],
            labels=['Female', 'Male'],
            colors=['pink', 'lightblue'],
            autopct='%1.1f%%', startangle=90)
    ax3.set_title('Non-Technical Occupations: Gender Ratio')

    plt.tight_layout()
    st.pyplot(fig)


    with st.expander("📊 Click to view the categorized summary data table"):
        st.dataframe(category_stats.round(3),hide_index=True)

    with st.expander("📋 Click to view detailed occupational classifications"):
        for category in category_stats['Category'].unique():
            st.markdown(f"**{category} Occupations:**")
            category_data = plot_df[plot_df['Category'] == category]
            st.dataframe(category_data[['Occupation', 'Female', 'Male']].round(2),hide_index=True)

    # Chart 4
    st.markdown('---')
    st.subheader("📈 Female Ratio Trends in Technical vs Non-Technical Occupations (Over Time)")

    year_columns = [col for col in female_df.columns if col.strip().isdigit() and len(col.strip()) == 4]
    year_columns.sort()

    years = []
    tech_female_ratios = []
    nontech_female_ratios = []

    for year in year_columns:
        female_data_year = female_df.iloc[1:10, female_df.columns.get_loc(year)].astype(float).reset_index(drop=True)
        male_data_year = male_df.iloc[1:10, male_df.columns.get_loc(year)].astype(float).reset_index(drop=True)

        temp_df = pd.DataFrame({
            'Occupation': occupations2,
            'Female': female_data_year,
            'Male': male_data_year
        })
        temp_df['Category'] = temp_df['Occupation'].apply(
            lambda x: 'Technical' if x in technical_occupations else 'Non-Technical'
        )

        category_stats_year = temp_df.groupby('Category').agg({
            'Female': 'sum',
            'Male': 'sum'
        }).reset_index()

        category_stats_year['Total'] = category_stats_year['Female'] + category_stats_year['Male']
        category_stats_year['Female_Ratio'] = category_stats_year['Female'] / category_stats_year['Total']

        tech_ratio = category_stats_year[category_stats_year['Category'] == 'Technical']['Female_Ratio'].values
        nontech_ratio = category_stats_year[category_stats_year['Category'] == 'Non-Technical']['Female_Ratio'].values

        if len(tech_ratio) > 0 and len(nontech_ratio) > 0:
            years.append(int(year))
            tech_female_ratios.append(tech_ratio[0])
            nontech_female_ratios.append(nontech_ratio[0])

    trend_df = pd.DataFrame({
        "Year": years,
        "Technical": tech_female_ratios,
        "Non-Technical": nontech_female_ratios
    })

    trend_long = trend_df.melt(id_vars="Year", var_name="Category", value_name="Female Ratio")

    fig_trend = px.line(
        trend_long,
        x="Year",
        y="Female Ratio",
        color="Category",
        markers=True,
        color_discrete_map={
            "Technical": "blue",
            "Non-Technical": "red"
        },
        template="plotly_white",
        title=''
    )

    fig_trend.update_traces(
        hovertemplate="<b>%{x}</b><br>%{y:.1%}<br>%{fullData.name}",
        line=dict(width=3),
        marker=dict(size=8)
    )
    fig_trend.update_layout(
        yaxis_tickformat=".0%",
        yaxis_title="Female Ratio",
        xaxis_title="Year",
        title_x=0.5,
        height=400,
        width=900,
        title=''
    )

    # st.plotly_chart(fig_trend, use_container_width=False)
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.plotly_chart(fig_trend, use_container_width=False)


    # st.subheader('📈 Long-term Trends')
    st.markdown('---')
    col1,col2= st.columns(2)
    with col1:
        st.subheader('🏭 Industry Trends')
        # compute female prop
        df_total=df_concat.groupby(['Industry','Year'])['Employment'].sum().reset_index()
        df_total.rename({'Employment':'Total Employment'},axis=1,inplace=True)
        df_props=pd.merge(df_concat,df_total,on=['Industry','Year'])
        df_props['Prop.']=df_props['Employment']/df_props['Total Employment']*100
        # line charts by industry
        fig=px.line(df_props[df_props['Gender']=='female'],
                    x='Year',
                    y='Prop.',
                    line_group='Industry',
                    color='Industry',
                    markers=True,
                    hover_data={'Prop.':':.1f'},
                    title='Female share in Employed Residents by Industry (2000~2004)')
        fig.update_layout(yaxis_title='Prop.(%)',
                        title_x=0.06)
        st.plotly_chart(fig,use_container_width=True)
        with st.expander('View Data'):
            df_show=df_props[df_props['Gender']=='female'][['Year','Industry','Prop.']]
            df_show=df_show.reset_index(drop=True)
            df_show.rename({'Prop.':'Female Prop.'},axis=1,inplace=True)
            st.dataframe(df_show.style.format({'Female Prop.':'{:.1f}'}))
    with col2:
        st.subheader("💼 Job Nature Trends")

        df_ft_long=transfer_long_table(df_ft)
        df_pt_long=transfer_long_table(df_pt)
        df_ft_long['Job Nature']='full-time'
        df_pt_long['Job Nature']='part-time'
        # concat full-time and part-time
        df_jobnature=pd.concat([df_ft_long,df_pt_long],ignore_index=True)

        fig=px.line(df_jobnature,
                    x='Year',
                    y='Prop.',
                    line_group='Job Nature',
                    color='Job Nature',
                    markers=True,
                    hover_data={'Prop.':':.1f'},
                    title='Female Share in Employed Residents by Job Nature (2009~2023)')
        fig.update_layout(yaxis_title='Prop.(%)',
                        title_x=0.06)
        st.plotly_chart(fig,use_container_width=True)
        with st.expander('View Data'):
            df_show=df_jobnature[['Year','Job Nature','Prop.']]
            df_show.rename({'Prop.':'Female Prop.'},axis=1,inplace=True)
            st.dataframe(df_show.style.format({'Female Prop.':'{:.1f}'}))

    st.markdown('---')
    st.subheader('🔍 Yearly Industrial Analysis')

    year_col1,year_col2,year_col3=st.columns([1,2,1])
    with year_col2:
        year=st.selectbox("📅 Select Year", 
                        [str(y) for y in range(2024,1999,-1)],
                        index=0,
                        help="Choose a year to see detailed analysis by gender and industry")
    col3,col4=st.columns(2)
    with col3:
        fig=px.bar(df_props[df_props['Year']==year],
                    x='Industry',
                    y='Employment',
                    color='Gender',
                    barmode='stack',
                    text='Employment',
                    text_auto=".1f",
                    hover_data={'Employment':':.1f'},
                    title=f'Employed Residents by Gender and Industry ({year})')
        fig.update_layout(yaxis_title='Employment (thousand)',
                        title_x=0.16)
        st.plotly_chart(fig,use_container_width=True)
    with col4:         
        # pie charts by industry
        fig = px.pie(df_props[df_props['Year']==year],
                    names="Gender",
                    values="Prop.",
                    facet_col="Industry",           
                    facet_col_wrap=2,               
                    color="Gender",
                    # title=f'Gender Composition by Industry ({year})'
                    )
        # adjust format
        fig.update_traces(textinfo='percent+label', pull=[0.05, 0])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
else:
    st.title("Employment Insights by Age 👶🧓")
    DATA_PATH = Path("M182171-table.csv")
    # Chart 1: Average Age of Employed Residents

    st.header("Chart 1: Average Age of Employed Residents (2000–2024)")

    @st.cache_data
    def load_and_compute_avg_age(path: Path):
        """Load CSV, clean it, and calculate weighted average age by year."""
        raw=pd.read_csv(path, skiprows=10)
        raw=raw.dropna(axis=1, how='all')
        raw=raw.rename(columns={raw.columns[0]: "Label"})
        raw=raw[raw["Label"].notna()]
        raw=raw[~raw["Label"].str.contains("Definitions|Footnotes|Notes|Notation", na=False)]

        is_header=raw["Label"].str.startswith("Employed Residents Aged") | raw["Label"].str.startswith("All Occupation Groups")
        raw["is_header"] = is_header

        current=None
        age_groups = []
        for lbl, is_h in zip(raw["Label"], raw["is_header"]):
            if is_h:
                if lbl.startswith("All Occupation Groups"):
                    current = "All Ages"
                else:
                    current=lbl.replace("Employed Residents ", "").strip()
            age_groups.append(current)
        raw["AgeGroup"] = age_groups
        raw["Occupation"] = raw["Label"].str.strip()

        occ_rows = raw[~raw["is_header"]].copy()
        long = occ_rows.melt(id_vars=["AgeGroup","Occupation"], var_name="Year", value_name="Employment")
        long["Year"] = pd.to_numeric(long["Year"], errors="coerce")
        long["Employment"] = pd.to_numeric(long["Employment"], errors="coerce")
        long = long.dropna(subset=["Year","Employment"])

        totals = long.groupby(["Year","AgeGroup"], as_index=False)["Employment"].sum()
        totals = totals[totals["AgeGroup"] != "All Ages"]

        age_mid = {
            'Aged 15 - 19 Years': 17,
            'Aged 20 - 24 Years': 22,
            'Aged 25 - 29 Years': 27,
            'Aged 30 - 34 Years': 32,
            'Aged 35 - 39 Years': 37,
            'Aged 40 - 44 Years': 42,
            'Aged 45 - 49 Years': 47,
            'Aged 50 - 54 Years': 52,
            'Aged 55 - 59 Years': 57,
            'Aged 60 - 64 Years': 62,
            'Aged 65 Years & Over': 67
        }
        totals["AgeMid"] = totals["AgeGroup"].map(age_mid)

        avg_age = (totals.groupby("Year")
                .apply(lambda g: (g["Employment"] * g["AgeMid"]).sum() / g["Employment"].sum())
                .reset_index(name="AverageAge"))
        return avg_age

    avg_age = load_and_compute_avg_age(DATA_PATH)

    # Two-column layout: left table, right chart
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Average Age by Year")
        st.dataframe(avg_age, use_container_width=True)
        csv_buffer = StringIO()
        avg_age.to_csv(csv_buffer, index=False)
        st.download_button("⬇️ Download CSV", csv_buffer.getvalue(), "average_age.csv", "text/csv")

    with col2:
        fig1 = px.line(
            avg_age,
            x="Year",
            y="AverageAge",
            markers=True,
            title="Average Age of Employed Residents (2000–2024)",
            labels={"AverageAge": "Average Age (years)", "Year": "Year"}
        )
        fig1.update_traces(hovertemplate="Year %{x}<br>Average Age: %{y:.2f}")
        fig1.update_layout(hovermode="x unified", 
                           template="plotly_white",
                           title_x=0.1)
        st.plotly_chart(fig1, use_container_width=True)

    # Chart 2: Total Employed Residents by Age Group

    st.header("Chart 2: Total Employed Residents by Age Group (2000–2024)")

    @st.cache_data
    def load_employment_by_agegroup(path: Path):
        """Load CSV, clean it, and aggregate employment by age group and year."""
        raw = pd.read_csv(path, skiprows=10)
        raw = raw.dropna(axis=1, how='all')
        raw = raw.rename(columns={raw.columns[0]: "Label"})
        raw = raw[raw["Label"].notna()]
        raw = raw[~raw["Label"].str.contains("Definitions|Footnotes|Notes|Notation", na=False)]

        is_header = raw["Label"].str.startswith("Employed Residents Aged") | raw["Label"].str.startswith("All Occupation Groups")
        raw["is_header"] = is_header

        current = None
        age_groups = []
        for lbl, is_h in zip(raw["Label"], raw["is_header"]):
            if is_h:
                if lbl.startswith("All Occupation Groups"):
                    current = "All Ages"
                else:
                    current = lbl.replace("Employed Residents ", "").strip()
            age_groups.append(current)
        raw["AgeGroup"] = age_groups
        raw["Occupation"] = raw["Label"].str.strip()

        occ_rows = raw[~raw["is_header"]].copy()
        long = occ_rows.melt(id_vars=["AgeGroup","Occupation"], var_name="Year", value_name="Employment")
        long["Year"] = pd.to_numeric(long["Year"], errors="coerce")
        long["Employment"] = pd.to_numeric(long["Employment"], errors="coerce")
        long = long.dropna(subset=["Year","Employment"])

        totals = long.groupby(["Year","AgeGroup"], as_index=False)["Employment"].sum()
        totals = totals[totals["AgeGroup"] != "All Ages"]

        return totals

    totals = load_employment_by_agegroup(DATA_PATH)

    # Two-column layout again: left table, right chart
    col3, col4 = st.columns([1, 2])

    with col3:
        st.subheader("Employment by Age Group and Year")
        pivot = totals.pivot(index="Year", columns="AgeGroup", values="Employment").sort_index()
        st.dataframe(pivot, use_container_width=True)
        csv_buffer2 = StringIO()
        pivot.to_csv(csv_buffer2)
        st.download_button("⬇️ Download CSV", csv_buffer2.getvalue(), "employment_by_agegroup.csv", "text/csv")

    with col4:
        # Custom red-blue color system
        color_palette = [
            "#003f5c", "#2f4b7c", "#665191", "#a05195", "#d45087",
            "#f95d6a", "#ff7c43", "#ffa600"
        ]
        fig2 = px.area(
            totals,
            x="Year",
            y="Employment",
            color="AgeGroup",
            title="Total Employed Residents by Age Group (2000–2024)",
            labels={"Employment": "Employment (Thousands)", "Year": "Year", "AgeGroup": "Age Group"},
            color_discrete_sequence=color_palette
        )
        fig2.update_traces(hovertemplate="Year %{x}<br>%{fullData.name}: %{y:,.0f}")
        fig2.update_layout(hovermode="x unified", 
                           template="plotly_white", 
                           legend_title="Age Group",
                           title_x=0.1)
        st.plotly_chart(fig2, use_container_width=True)


    # ===== Title =====
    st.markdown("<h3 style='margin:0'>Chart 3: Employment by Occupation & Age</h3>", unsafe_allow_html=True)

    # ===== Data Loading =====
    DATA_PATH = Path("M182171-table.csv")

    @st.cache_data(show_spinner=False)
    def load_long_df(path: str):
        raw = pd.read_csv(path, skiprows=10)
        raw = raw.dropna(axis=1, how='all')
        raw = raw.rename(columns={raw.columns[0]: "Label"})
        raw = raw[raw["Label"].notna()]
        raw = raw[~raw["Label"].str.contains("Definitions|Footnotes", na=False)]

        is_header = raw["Label"].str.startswith("Employed Residents Aged") | raw["Label"].str.startswith("All Occupation Groups")
        raw["is_header"] = is_header

        current = None
        age_groups = []
        for lbl, is_h in zip(raw["Label"], raw["is_header"]):
            if is_h:
                if lbl.startswith("All Occupation Groups"):
                    current = "All Ages"
                else:
                    current = lbl.replace("Employed Residents ", "").strip()
            age_groups.append(current)

        raw["AgeGroup"] = age_groups
        raw["Occupation"] = raw["Label"].str.strip()

        rows = raw[~raw["is_header"]].copy()

        value_cols = [c for c in rows.columns if c not in ["Label","is_header","AgeGroup","Occupation"]]
        long = rows.melt(id_vars=["AgeGroup","Occupation"], var_name="Year", value_name="Employment")
        long["Year"] = pd.to_numeric(long["Year"], errors="coerce").round().astype("Int64")
        long["Employment"] = pd.to_numeric(long["Employment"], errors="coerce")
        long = long.dropna(subset=["Year","Employment"])
        return long

    try:
        long = load_long_df(DATA_PATH)
    except Exception as e:
        st.error(f"Failed to load: {DATA_PATH}\n\nError: {e}")
        st.stop()

    # ===== Persistent Occupation Filter =====
    all_occs = sorted(long["Occupation"].unique())
    default_key_occs = [
        "Managers & Administrators (Including Working Proprietors)",
        "Professionals",
        "Clerical Support Workers",
        "Service & Sales Workers",
    ]
    default_key_occs = [o for o in default_key_occs if o in all_occs] or all_occs[:4]

    if "key_occs" not in st.session_state:
        st.session_state.key_occs = default_key_occs
    st.markdown("**Occupations**")
    st.multiselect(
        "Select occupations to display",
        options=all_occs,
        default=st.session_state.key_occs,
        key="key_occs",
        help="Your selection persists across years."
    )
    sel_occs = st.session_state.key_occs

    # ===== Build Plotly native animation (smooth) =====
    age_order = [
        'Aged 15 - 19 Years','Aged 20 - 24 Years','Aged 25 - 29 Years','Aged 30 - 34 Years',
        'Aged 35 - 39 Years','Aged 40 - 44 Years','Aged 45 - 49 Years','Aged 50 - 54 Years',
        'Aged 55 - 59 Years','Aged 60 - 64 Years','Aged 65 Years & Over'
    ]

    df_anim = long[(long["Occupation"].isin(sel_occs)) & (long["AgeGroup"] != "All Ages")].copy()
    df_anim["AgeGroup"] = pd.Categorical(df_anim["AgeGroup"], categories=age_order, ordered=True)
    df_anim = df_anim.sort_values(["Year","AgeGroup","Occupation"])

    years = sorted([int(y) for y in df_anim["Year"].unique()])

    # Initial (latest)
    init_year = years[-1]
    pivot0 = (df_anim[df_anim["Year"] == init_year]
            .pivot(index="AgeGroup", columns="Occupation", values="Employment")
            .reindex(age_order))

    fig = go.Figure()
    for occ in sel_occs:
        yvals = pivot0[occ].tolist() if occ in pivot0.columns else [None]*len(age_order)
        fig.add_bar(
            name=str(occ),
            x=age_order,
            y=yvals,
            hovertemplate="<b>%{x}</b><br>Occupation: " + str(occ) + "<br>Employment: %{y:,.2f} (thousands)<extra></extra>"
        )

    frames = []
    for yr in years:
        pvt = (df_anim[df_anim["Year"] == yr]
            .pivot(index="AgeGroup", columns="Occupation", values="Employment")
            .reindex(age_order))
        data = []
        for occ in sel_occs:
            yvals = pvt[occ].tolist() if occ in pvt.columns else [None]*len(age_order)
            data.append(go.Bar(x=age_order, y=yvals, name=str(occ)))
        frames.append(go.Frame(data=data, name=str(int(yr))))
    fig.frames = frames

    # Slider + Play/Pause (moved below, set negative y)
    fig.update_layout(
        barmode="group",
        height=420,
        margin=dict(l=10, r=10, t=30, b=150),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="Age Group",
        yaxis_title="Employment (Thousands)",
        sliders=[
            dict(
                active=len(years)-1,
                x=0.0, y=-0.10, xanchor="left", yanchor="top",
                len=1.0,
                pad=dict(t=20, b=10),
                currentvalue=dict(visible=True, prefix="Year: ", xanchor="right", offset=10),
                transition={"duration": 300, "easing": "cubic-in-out"},
                steps=[
                    dict(method="animate",
                        args=[[str(int(yr))],
                            {"frame": {"duration": 550, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 300, "easing": "cubic-in-out"}}],
                        label=str(int(yr)))
                    for yr in years
                ]
            )
        ],
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.0, y=-0.18, xanchor="left", yanchor="top",
                buttons=[
                    dict(label="▶ Play", method="animate",
                        args=[None, {"frame": {"duration": 550, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 300, "easing": "cubic-in-out"}}]),
                    dict(label="⏸ Pause", method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}}])
                ]
            )
        ]
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ===== Collapsible table (defaults to latest year; offer a quick sync button) =====
    # We can't read the current animated slider value on the Python side, but we can let the user sync it with one click.
    if "table_year" not in st.session_state:
        st.session_state.table_year = years[-1]


    with st.expander("Top occupation by age group (share within age group)", expanded=False):
        table_years = sorted([int(y) for y in long["Year"].dropna().unique()])
        table_year = st.selectbox("Table year", options=table_years, index=len(table_years)-1, help="Choose a year for the table")
        df_table = long[(long["Year"] == table_year) & (long["AgeGroup"] != "All Ages")].copy()
        totals = df_table.groupby("AgeGroup")["Employment"].transform("sum")
        df_table["Share"] = df_table["Employment"] / totals
        idx = df_table.groupby("AgeGroup")["Share"].idxmax()
        top_by_age = (df_table.loc[idx, ["AgeGroup","Occupation","Employment","Share"]]
                    .sort_values("AgeGroup"))
        top_by_age["Employment"] = top_by_age["Employment"].map(lambda v: f"{v:,.2f}")
        top_by_age["Share"] = top_by_age["Share"].map(lambda v: f"{v:.1%}")
        st.dataframe(top_by_age.reset_index(drop=True), use_container_width=True)

    # ===== Downloads =====
    d1, d2, _ = st.columns([0.25, 0.35, 1])
    try:
        import plotly.io as pio
        png_bytes = pio.to_image(fig, format="png", width=1300, height=520, scale=2)
        d1.download_button("Download chart PNG", data=png_bytes, file_name="age_occupation_animated.png", mime="image/png")
    except Exception:
        d1.caption("Install kaleido to enable chart PNG download")
    # Export latest-year filtered data for selected occupations
    latest_year = years[-1]
    df_latest = long[(long["Year"] == latest_year) & (long["AgeGroup"] != "All Ages") & (long["Occupation"].isin(sel_occs))]
    csv_buf = io.StringIO()
    df_latest.to_csv(csv_buf, index=False)
    d2.download_button("Download latest-year data CSV", data=csv_buf.getvalue(),
                    file_name=f"employment_{int(latest_year)}_filtered.csv", mime="text/csv")

    st.caption(f"File: {DATA_PATH}")
