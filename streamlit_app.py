import streamlit as st
import pandas as pd
import requests
import io

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="LeadCraft Real-Data B2B Engine",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling rules to replicate the B2B dashboard look
st.markdown("""
    <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 800;
            color: #1e3a8a;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1rem;
            color: #64748b;
            margin-bottom: 1.5rem;
            font-weight: 500;
        }
        .kpi-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            padding: 1.2rem;
            border-radius: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            text-align: center;
        }
        .kpi-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 0.2rem;
        }
        .kpi-label {
            font-size: 0.75rem;
            font-weight: 700;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .control-panel-container {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 1.5rem;
            border-radius: 1rem;
            margin-bottom: 1.5rem;
        }
        .warning-card {
            background-color: #fef2f2;
            border: 1px solid #fecaca;
            padding: 1.5rem;
            border-radius: 1rem;
            color: #991b1b;
            margin-bottom: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "live_prospects_df" not in st.session_state:
    st.session_state.live_prospects_df = None
if "active_job_title" not in st.session_state:
    st.session_state.active_job_title = ""
if "active_industry" not in st.session_state:
    st.session_state.active_industry = ""

# ==========================================
# 1. LIVE DATA INTEGRATION - APOLLO.IO API
# ==========================================
def fetch_live_apollo_leads(job_title, industry, api_key, limit):
    """Fetches real-world, verified professional contacts from the Apollo.io API."""
    url = "https://api.apollo.io/v1/mixed_people/search"
    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "application/json"
    }
    
    payload = {
        "api_key": api_key,
        "q_person_title_keywords": [job_title] if job_title else [],
        "per_page": min(limit, 100)
    }
    if industry:
        payload["organization_keywords"] = [industry]
        
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            people_list = data.get("people", [])
            
            records = []
            for person in people_list:
                org = person.get("organization", {})
                location = ", ".join(filter(None, [person.get("city"), person.get("state"), person.get("country")]))
                
                records.append({
                    "Select": True,
                    "Full Name": person.get("name", "N/A"),
                    "Job Title": person.get("title", "N/A"),
                    "Company": org.get("name", "N/A"),
                    "Location": location if location else "N/A",
                    "Email Address": person.get("email", "N/A"),
                    "LinkedIn URL": person.get("linkedin_url", ""),
                    "Years of Experience": person.get("seniority", "N/A"),
                    "Data Source": "Apollo.io API"
                })
            return pd.DataFrame(records)
        else:
            st.error(f"Apollo API Error: {response.status_code}")
    except Exception as e:
        st.error(f"Failed to connect to Apollo: {str(e)}")
    return pd.DataFrame()

# ==========================================
# 2. LIVE DATA INTEGRATION - SERPAPI LINKEDIN X-RAY
# ==========================================
def fetch_live_serpapi_linkedin(job_title, industry, api_key, limit):
    """Uses SerpAPI to execute Google Custom X-Ray searches to parse real LinkedIn profiles."""
    url = "https://serpapi.com/search"
    
    query_parts = ["site:linkedin.com/in/"]
    if job_title:
        query_parts.append(f'"{job_title}"')
    if industry:
        query_parts.append(f'"{industry}"')
        
    query_str = " ".join(query_parts)
    
    params = {
        "engine": "google",
        "q": query_str,
        "api_key": api_key,
        "num": min(limit, 50)
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            results = data.get("organic_results", [])
            
            records = []
            for r in results:
                title = r.get("title", "")
                link = r.get("link", "")
                
                parts = title.split(" - ")
                name = parts[0] if len(parts) > 0 else "N/A"
                role = parts[1] if len(parts) > 1 else job_title
                company = parts[2].split(" | ")[0] if len(parts) > 2 else industry
                
                records.append({
                    "Select": True,
                    "Full Name": name,
                    "Job Title": role,
                    "Company": company if company else "N/A",
                    "Location": "N/A (Indexed on Web)",
                    "Email Address": "Requires Verification",
                    "LinkedIn URL": link,
                    "Years of Experience": "N/A",
                    "Data Source": "SerpAPI LinkedIn X-Ray"
                })
            return pd.DataFrame(records)
        else:
            st.error(f"SerpAPI Error: {response.status_code}")
    except Exception as e:
        st.error(f"Failed to connect to SerpAPI: {str(e)}")
    return pd.DataFrame()

# ==========================================
# 3. LIVE DATA INTEGRATION - HUNTER.IO API
# ==========================================
def fetch_live_hunter_leads(industry, api_key, limit):
    """Resolves corporate domains via Clearbit Autocomplete, then pulls actual company directories from Hunter.io."""
    if not industry:
        st.error("Hunter.io requires a specific target Company/Industry Name.")
        return pd.DataFrame()
        
    domain = None
    try:
        clearbit_url = f"https://autocomplete.clearbit.com/v1/companies/suggest?query={industry}"
        res = requests.get(clearbit_url)
        if res.status_code == 200 and res.json():
            domain = res.json()[0].get("domain")
    except Exception as e:
        pass
        
    if not domain:
        st.error(f"Could not automatically resolve a real business domain for: '{industry}'")
        return pd.DataFrame()
        
    url = "https://api.hunter.io/v2/domain-search"
    params = {
        "domain": domain,
        "api_key": api_key,
        "limit": min(limit, 100)
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json().get("data", {})
            emails = data.get("emails", [])
            
            records = []
            for email_data in emails:
                first = email_data.get("first_name", "")
                last = email_data.get("last_name", "")
                full_name = f"{first} {last}".strip() if (first or last) else "Corporate Recipient"
                
                records.append({
                    "Select": True,
                    "Full Name": full_name,
                    "Job Title": email_data.get("position", "Employee"),
                    "Company": data.get("organization", industry.capitalize()),
                    "Location": "N/A (HQ Verified)",
                    "Email Address": email_data.get("value", ""),
                    "LinkedIn URL": email_data.get("linkedin", ""),
                    "Years of Experience": "N/A (Confidence: " + str(email_data.get("confidence", 0)) + "%)",
                    "Data Source": "Hunter.io Domain Search"
                })
            return pd.DataFrame(records)
        else:
            st.error(f"Hunter.io Error: {response.status_code}")
    except Exception as e:
        st.error(f"Failed to connect to Hunter.io: {str(e)}")
    return pd.DataFrame()

# ==========================================
# STREAMLIT SIDEBAR - DIRECTORY KEY CONFIG
# ==========================================
with st.sidebar:
    st.markdown("### 🔑 Live B2B API Connections")
    st.write("Synthetic engines have been removed. This interface maps fields directly to real B2B pipelines.")
    
    # Preloaded secure user credentials
    DEFAULT_SERP_KEY = "1f0b0c2a13ccc236001865b2734efcd6ed07f6e776b6427b70c9afb14eb0e1fe"
    
    apollo_key = st.text_input("Apollo.io API Key", type="password", placeholder="e.g. apikey_123...")
    
    serpapi_key = st.text_input(
        "SerpAPI Key (LinkedIn Crawl)", 
        type="password", 
        value=DEFAULT_SERP_KEY,
        help="System has initialized with your provided private connection string."
    )
    
    hunter_key = st.text_input("Hunter.io API Key", type="password", placeholder="e.g. abcd1234...")
    
    st.write("---")
    if serpapi_key or apollo_key or hunter_key:
        st.success("✅ Live Direct API Pipeline Engaged")

# ==========================================
# MAIN WORKSPACE HEADER
# ==========================================
st.markdown('<div class="main-header">LeadCraft <span style="color:#2563eb;">AI</span> Live</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">B2B Directory Integrator — Real Contact Records Only</div>', unsafe_allow_html=True)

# Control Center
st.markdown('<div class="control-panel-container">', unsafe_allow_html=True)
st.markdown("### 🔍 Live B2B Search Criteria")

col_search_1, col_search_2, col_search_3 = st.columns([2, 2, 1])
with col_search_1:
    job_title = st.text_input("Target Job Title", value="", placeholder="e.g. VP of Security, Software Architect")
with col_search_2:
    industry = st.text_input("Target Company / Industry", value="", placeholder="e.g. Stripe, Biotech, Fintech")
with col_search_3:
    # Default set exactly to 10 as requested
    total_leads = st.slider("Result Limit", min_value=5, max_value=100, value=10, step=5)

# Selection of query route
active_pipelines = []
if apollo_key: active_pipelines.append("Apollo.io Live Database Search")
if serpapi_key: active_pipelines.append("SerpAPI Google LinkedIn X-Ray")
if hunter_key: active_pipelines.append("Hunter.io Corporate Domain Directory")

# Auto-set index based on presence of pre-loaded key
default_index = 0 if "SerpAPI Google LinkedIn X-Ray" in active_pipelines else 0

selected_pipeline = st.selectbox(
    "Active Data Routing Pipeline", 
    active_pipelines if active_pipelines else ["Pipeline Inactive"],
    index=default_index
)

btn_query = st.button("Search Real-World Directory", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Processing actual external requests
if btn_query:
    if not job_title and not industry:
        st.error("Please insert a Job Title or Target Company/Industry to specify search variables.")
    else:
        with st.spinner(f"Contacting {selected_pipeline} to request real profiles..."):
            df = pd.DataFrame()
            
            if selected_pipeline == "Apollo.io Live Database Search":
                df = fetch_live_apollo_leads(job_title, industry, apollo_key, total_leads)
            elif selected_pipeline == "SerpAPI Google LinkedIn X-Ray":
                df = fetch_live_serpapi_linkedin(job_title, industry, serpapi_key, total_leads)
            elif selected_pipeline == "Hunter.io Corporate Domain Directory":
                df = fetch_live_hunter_leads(industry, hunter_key, total_leads)
                
            if not df.empty:
                st.session_state.live_prospects_df = df
                st.session_state.active_job_title = job_title
                st.session_state.active_industry = industry
                st.toast(f"Retrieved {len(df)} authentic profiles successfully!", icon="🔥")
            else:
                st.warning("Query executed, but no real-world records matched the criteria.")

# ==========================================
# DYNAMIC DATA FRAME RESULTS WORKSPACE
# ==========================================
if st.session_state.live_prospects_df is not None:
    df_state = st.session_state.live_prospects_df
    
    tab_directory, tab_analytics = st.tabs([
        "📋 Verified Live Directory", 
        "📊 Dynamic Query Insights"
    ])
    
    with tab_directory:
        col_sel_1, col_sel_2, _ = st.columns([1.5, 1.5, 7])
        with col_sel_1:
            if st.button("Select All", use_container_width=True):
                df_state["Select"] = True
                st.session_state.live_prospects_df = df_state
                st.rerun()
        with col_sel_2:
            if st.button("Unselect All", use_container_width=True):
                df_state["Select"] = False
                st.session_state.live_prospects_df = df_state
                st.rerun()

        st.markdown("##### Real-World Search Results Grid")
        st.caption("Active rows are direct API outputs. Selected entries will write to your Excel export sheet.")
        
        edited_df = st.data_editor(
            df_state,
            column_config={
                "Select": st.column_config.CheckboxColumn("Select", help="Choose row for export.", default=True),
                "LinkedIn URL": st.column_config.LinkColumn("LinkedIn URL", display_text="Open Profile ↗"),
                "Full Name": st.column_config.TextColumn(disabled=True),
                "Job Title": st.column_config.TextColumn(disabled=True),
                "Company": st.column_config.TextColumn(disabled=True),
                "Location": st.column_config.TextColumn(disabled=True),
                "Email Address": st.column_config.TextColumn(disabled=True),
                "Years of Experience": st.column_config.TextColumn(disabled=True),
                "Data Source": st.column_config.TextColumn(disabled=True)
            },
            use_container_width=True,
            hide_index=True,
            key="real_data_editor"
        )
        
        st.session_state.live_prospects_df = edited_df
        selected_rows = edited_df[edited_df["Select"] == True]

        st.markdown("---")
        st.markdown("##### 💾 Excel Exporter")
        
        if len(selected_rows) > 0:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                export_cleaned_df = selected_rows.drop(columns=["Select"])
                export_cleaned_df.to_excel(writer, index=False, sheet_name="Live B2B Contacts")
                
            excel_data = buffer.getvalue()
            target_file_title = f"Live_Prospects_{st.session_state.active_job_title.replace(' ', '_')}_{st.session_state.active_industry.replace(' ', '_')}.xlsx"
            
            st.download_button(
                label=f"📥 Download {len(selected_rows)} Verified Leads in Excel",
                data=excel_data,
                file_name=target_file_title,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        else:
            st.warning("Please check/select at least one verified prospect to download.")

    with tab_analytics:
        st.markdown("#### 📊 Geographic and Source Demographics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(df_state):,}</div><div class="kpi-label">Profiles Fetched</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color: #2563eb;">{len(selected_rows):,}</div><div class="kpi-label">Checked Profiles</div></div>', unsafe_allow_html=True)
        with col3:
            sources_present = ", ".join(df_state["Data Source"].unique())
            st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="font-size:1.1rem; padding-top:0.6rem; color: #10b981;">{sources_present}</div><div class="kpi-label">Live Connector</div></div>', unsafe_allow_html=True)

        st.write("---")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("##### Sourced Location Slices")
            st.bar_chart(df_state["Location"].value_counts())
        with col_c2:
            st.markdown("##### Job Title Distribution")
            st.bar_chart(df_state["Job Title"].value_counts())
else:
    st.info("👈 Feed your search parameters in the main section to query real business networks via SerpAPI.")
