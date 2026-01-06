import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Kabaddi Team Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ---------------- READ DATA ----------------
@st.cache_data
def load_match_data(match_file):
    match_sheets = pd.read_excel(match_file, sheet_name=None)
    cleaned_sheets = []
    
    for name, df in match_sheets.items():
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty:
            continue
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
        df["Tournament"] = str(name)
        cleaned_sheets.append(df)
    
    matches = pd.concat(cleaned_sheets, ignore_index=True)
    matches.columns = matches.columns.astype(str)
    
    # Detect player name column
    possible_name_cols = [c for c in matches.columns if "name" in c.lower() or "player" in c.lower()]
    if possible_name_cols:
        matches.rename(columns={possible_name_cols[0]: "Player"}, inplace=True)
    else:
        st.error("❌ Could not find a column with player names!")
        st.stop()
    
    # Convert numeric data
    for col in matches.columns:
        if col not in ["Player", "Tournament"]:
            matches[col] = pd.to_numeric(matches[col], errors="coerce")
    
    # Compute total points
    matches["Total Points"] = matches.select_dtypes("number").sum(axis=1)
    
    return matches, cleaned_sheets

@st.cache_data
def load_attendance_data(attendance_file):
    try:
        att_xls = pd.ExcelFile(attendance_file)
        att_frames = []
        
        for sh in att_xls.sheet_names:
            df_sh = pd.read_excel(attendance_file, sheet_name=sh)
            df_sh = df_sh.dropna(how="all").dropna(axis=1, how="all")
            df_sh["__source_sheet"] = sh
            att_frames.append(df_sh)
        
        if att_frames:
            att_all = pd.concat(att_frames, ignore_index=True, sort=False)
        else:
            att_all = pd.DataFrame()
        
        # Detect player column
        att_player_col = None
        for c in att_all.columns:
            if isinstance(c, str) and any(tok in c.lower() for tok in ["name", "player", "student"]):
                att_player_col = c
                break
        
        # Detect attendance value columns
        present_col = None
        total_col = None
        
        for c in att_all.columns:
            if isinstance(c, str):
                if "present" in c.lower():
                    present_col = c
                elif "practic" in c.lower() or "total" in c.lower() or "section" in c.lower():
                    total_col = c
        
        if att_player_col and present_col and total_col:
            att_clean = att_all[[att_player_col, present_col, total_col]].copy()
            att_clean.columns = ["Player", "Present", "Total"]
            
            # Clean player names
            att_clean["Player"] = att_clean["Player"].astype(str).str.strip().str.upper()
            att_clean = att_clean[att_clean["Player"] != "NAN"]
            
            # Calculate attendance percentage
            att_clean["Present"] = pd.to_numeric(att_clean["Present"], errors="coerce")
            att_clean["Total"] = pd.to_numeric(att_clean["Total"], errors="coerce")
            att_clean["Attendance_pct"] = (att_clean["Present"] / att_clean["Total"] * 100).round(2)
            
            # Group by player and take mean
            att_summary = att_clean.groupby("Player").agg({
                "Attendance_pct": "mean",
                "Present": "sum",
                "Total": "sum"
            }).reset_index()
            
            return att_summary
        
        return pd.DataFrame()
    
    except Exception as e:
        st.error(f"Error reading attendance file: {e}")
        return pd.DataFrame()

# File paths
match_file = r"match point (kabbadi) final.xlsx"
attendance_file = r"Copy of Copy_of_Copy_of_PAX_-_Attendance_1111_processed_v2(1).xlsx"

# Load data
try:
    matches, cleaned_sheets = load_match_data(match_file)
    attendance = load_attendance_data(attendance_file)
    
    # Normalize player names and REMOVE NaN players
    matches["Player"] = matches["Player"].astype(str).str.strip().str.upper()
    matches = matches[matches["Player"].notna() & (matches["Player"] != "NAN") & (matches["Player"] != "")]
    
    # Merge attendance data
    if not attendance.empty:
        matches = matches.merge(attendance, on="Player", how="left")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.image("https://img.icons8.com/fluency/96/000000/sports.png", width=80)
st.sidebar.title("🏐 Kabaddi Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio("📊 Navigation", ["🏠 Overview", "👤 Individual Analysis", "🏆 Team Analysis", "📈 Attendance Insights"])

# =====================================================
# 🏠 OVERVIEW PAGE
# =====================================================
if page == "🏠 Overview":
    st.markdown('<div class="main-header">🏐 Kabaddi Team Dashboard</div>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Total Players", len(matches["Player"].unique()))
    with col2:
        st.metric("🏆 Tournaments", len(cleaned_sheets))
    with col3:
        st.metric("⚡ Total Points", int(matches["Total Points"].sum()))
    with col4:
        if not attendance.empty:
            st.metric("📊 Avg Attendance", f"{attendance['Attendance_pct'].mean():.1f}%")
        else:
            st.metric("📊 Avg Attendance", "N/A")
    
    st.markdown("---")
    
    # Two column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏅 Top 10 Players by Points")
        top_players = matches.groupby("Player")["Total Points"].sum()
        top_players = top_players[top_players > 0].sort_values(ascending=False).head(10)
        
        if len(top_players) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.Oranges(np.linspace(0.4, 0.8, len(top_players)))
            bars = ax.barh(range(len(top_players)), top_players.values, color=colors)
            ax.set_yticks(range(len(top_players)))
            ax.set_yticklabels(top_players.index)
            ax.set_xlabel("Total Points", fontsize=12, fontweight='bold')
            ax.set_title("Top 10 Players by Total Points", fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            
            for i, (bar, val) in enumerate(zip(bars, top_players.values)):
                ax.text(val + 0.5, i, f'{int(val)}', va='center', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("No valid player data available.")
    
    with col2:
        st.subheader("📊 Average Points per Player per Tournament")
        
        player_tournament_avg = matches.groupby("Player").agg({
            "Total Points": "sum",
            "Tournament": "nunique"
        }).reset_index()
        
        player_tournament_avg = player_tournament_avg[player_tournament_avg["Total Points"] > 0]
        
        player_tournament_avg["Avg Points per Tournament"] = (
            player_tournament_avg["Total Points"] / player_tournament_avg["Tournament"]
        )
        player_tournament_avg = player_tournament_avg.sort_values("Avg Points per Tournament", ascending=False).head(10)
        
        if len(player_tournament_avg) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(player_tournament_avg)))
            bars = ax.barh(range(len(player_tournament_avg)), 
                          player_tournament_avg["Avg Points per Tournament"].values, color=colors)
            ax.set_yticks(range(len(player_tournament_avg)))
            ax.set_yticklabels(player_tournament_avg["Player"])
            ax.set_xlabel("Average Points per Tournament", fontsize=12, fontweight='bold')
            ax.set_title("Top 10 Players by Average Performance", fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            
            for i, (bar, val) in enumerate(zip(bars, player_tournament_avg["Avg Points per Tournament"].values)):
                ax.text(val + 0.1, i, f'{val:.1f}', va='center', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("No valid player data available.")
    
    # Performance consistency and participation
    st.markdown("---")
    st.subheader("📈 Player Performance Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Performance Consistency (Std Dev vs Mean)")
        
        player_consistency = matches.groupby("Player")["Total Points"].agg(['mean', 'std', 'count']).reset_index()
        player_consistency = player_consistency[player_consistency['count'] >= 3]
        player_consistency = player_consistency.dropna(subset=['mean', 'std'])
        player_consistency = player_consistency[player_consistency['mean'] > 0]
        
        player_consistency['coefficient_of_variation'] = (player_consistency['std'] / player_consistency['mean']) * 100
        player_consistency = player_consistency.dropna()
        
        if len(player_consistency) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            scatter = ax.scatter(player_consistency['mean'], 
                               player_consistency['std'],
                               s=player_consistency['count'] * 50,
                               c=player_consistency['mean'],
                               cmap='RdYlGn',
                               alpha=0.6,
                               edgecolors='black',
                               linewidth=1.5)
            
            top_performers = player_consistency.nlargest(5, 'mean')
            for idx, row in top_performers.iterrows():
                ax.annotate(row['Player'], 
                           (row['mean'], row['std']),
                           xytext=(5, 5), 
                           textcoords="offset points", 
                           fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
            
            ax.set_xlabel("Mean Points per Tournament", fontsize=12, fontweight='bold')
            ax.set_ylabel("Standard Deviation", fontsize=12, fontweight='bold')
            ax.set_title("Player Consistency Analysis\n(Lower StdDev = More Consistent)", fontsize=13, fontweight='bold')
            ax.grid(alpha=0.3)
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Mean Points', fontsize=10)
            
            mean_avg = player_consistency['mean'].mean()
            std_avg = player_consistency['std'].mean()
            ax.axvline(mean_avg, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Avg Mean')
            ax.axhline(std_avg, color='blue', linestyle='--', alpha=0.5, linewidth=1.5, label='Avg StdDev')
            ax.legend(loc='upper right')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.caption("💡 *Best players*: High mean (right side) + Low std dev (bottom) = Consistent high performers")
            st.caption("📊 Bubble size represents number of tournaments played")
        else:
            st.warning("Not enough data for consistency analysis (need at least 3 tournaments per player).")
    
    with col2:
        st.markdown("#### 🏆 Tournament Participation & Performance")
        
        participation_data = matches.groupby("Player").agg({
            "Tournament": "nunique",
            "Total Points": ["sum", "mean"]
        }).reset_index()
        participation_data.columns = ["Player", "Tournaments_Played", "Total_Points", "Avg_Points"]
        
        participation_data = participation_data[participation_data["Total_Points"] > 0]
        participation_data = participation_data.sort_values("Tournaments_Played", ascending=False).head(15)
        
        if len(participation_data) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(len(participation_data))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, participation_data["Tournaments_Played"], width, 
                          label='Tournaments Played', color='#4ECDC4', edgecolor='black', linewidth=1)
            
            ax2 = ax.twinx()
            bars2 = ax2.bar(x + width/2, participation_data["Avg_Points"], width,
                           label='Avg Points/Tournament', color='#FF6B35', edgecolor='black', linewidth=1)
            
            ax.set_xlabel("Players", fontsize=12, fontweight='bold')
            ax.set_ylabel("Tournaments Played", fontsize=11, fontweight='bold', color='#4ECDC4')
            ax2.set_ylabel("Avg Points per Tournament", fontsize=11, fontweight='bold', color='#FF6B35')
            ax.set_title("Player Participation & Average Performance", fontsize=13, fontweight='bold')
            
            ax.set_xticks(x)
            ax.set_xticklabels(participation_data["Player"], rotation=45, ha='right', fontsize=9)
            
            ax.tick_params(axis='y', labelcolor='#4ECDC4')
            ax2.tick_params(axis='y', labelcolor='#FF6B35')
            
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.caption("💡 *Ideal players*: High participation (teal bars) + High average (orange bars)")
        else:
            st.warning("No valid participation data available.")

# =====================================================
# 👤 INDIVIDUAL ANALYSIS
# =====================================================
elif page == "👤 Individual Analysis":
    st.markdown('<div class="main-header">👤 Individual Player Analysis</div>', unsafe_allow_html=True)
    
    # Get valid players only (no NaN)
    valid_players = sorted([p for p in matches["Player"].unique() if p and str(p).upper() != "NAN"])
    
    if not valid_players:
        st.error("No valid player data available.")
        st.stop()
    
    selected_player = st.selectbox("🔍 Select a Player", valid_players, key="individual_player")
    
    player_df = matches[matches["Player"] == selected_player]
    total_points = player_df["Total Points"].sum()
    total_matches = player_df["Tournament"].nunique()
    avg_points = total_points / total_matches if total_matches > 0 else 0
    
    # Player metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Total Points", int(total_points))
    with col2:
        st.metric("🏆 Matches Played", int(total_matches))
    with col3:
        st.metric("📊 Avg Points/Match", f"{avg_points:.2f}")
    with col4:
        # IMPROVED ATTENDANCE DISPLAY
        if "Attendance_pct" in player_df.columns and not player_df["Attendance_pct"].isna().all():
            att_val = player_df["Attendance_pct"].iloc[0]
            st.metric("📅 Attendance", f"{att_val:.1f}%")
        else:
            # Calculate estimated attendance based on participation
            participation_rate = (total_matches / len(cleaned_sheets)) * 100
            st.metric("📅 Participation Rate", f"{participation_rate:.1f}%")
    
    st.markdown("---")
    
    # Performance analysis
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📈 {selected_player}'s Performance Across Tournaments")
        
        tournament_perf = player_df.groupby("Tournament")["Total Points"].sum().sort_values(ascending=False)
        tournament_perf = tournament_perf[~tournament_perf.index.str.contains("TOTAL|MATCHES", case=False, na=False)]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(tournament_perf)))
        bars = ax.bar(range(len(tournament_perf)), tournament_perf.values, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(tournament_perf)))
        ax.set_xticklabels([t[:25] + '...' if len(t) > 25 else t for t in tournament_perf.index], 
                          rotation=45, ha='right')
        ax.set_ylabel("Points Scored", fontsize=12, fontweight='bold')
        ax.set_title(f"{selected_player} - Tournament Performance", fontsize=14, fontweight='bold')
        
        for bar, val in zip(bars, tournament_perf.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(val)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("🏅 Player Rating")
        
        mean_all = matches.groupby("Player")["Total Points"].sum().mean()
        std_all = matches.groupby("Player")["Total Points"].sum().std()
        
        if total_points >= mean_all + std_all:
            rating = "⭐ EXCELLENT"
            color = "green"
            emoji = "🔥"
        elif total_points >= mean_all:
            rating = "✅ GOOD"
            color = "blue"
            emoji = "👍"
        else:
            rating = "⚪ MODERATE"
            color = "orange"
            emoji = "📊"
        
        player_rank = matches.groupby('Player')['Total Points'].sum().rank(ascending=False)[selected_player]
        
        st.markdown(f"""
        <div style='padding: 2rem; background-color: {color}20; border-radius: 10px; border-left: 5px solid {color};'>
            <h2 style='color: {color}; text-align: center;'>{emoji} {rating}</h2>
            <p style='text-align: center; font-size: 1.2rem;'>
                <strong>Total Points:</strong> {int(total_points)}<br>
                <strong>Team Average:</strong> {mean_all:.1f}<br>
                <strong>Rank:</strong> {player_rank:.0f} of {len(valid_players)}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Round-wise performance
        st.subheader("🎯 Round-wise Breakdown")
        round_cols = [col for col in player_df.columns 
                     if any(key in col.upper() for key in ["ROUND", "Q-FINAL", "QF", "SEMI", "FINAL"])]
        
        if round_cols:
            round_data = player_df[round_cols].sum()
            round_data = round_data.dropna()
            round_data = round_data[round_data > 0]
            round_data = round_data.sort_values(ascending=False)
            
            if len(round_data) > 0:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie(round_data.values, labels=round_data.index, autopct='%1.1f%%', 
                      startangle=90, colors=plt.cm.Set3.colors)
                ax.set_title("Points Distribution by Round", fontsize=12, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No round-wise data available for this player.")
    
    # Comparison with top players - REMOVE NaN
    st.markdown("---")
    st.subheader("📊 Comparison with Top 10 Players")
    
    top_10 = matches.groupby("Player")["Total Points"].sum().sort_values(ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ['#FF6B35' if p == selected_player else '#4ECDC4' for p in top_10.index]
    bars = ax.barh(range(len(top_10)), top_10.values, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_yticks(range(len(top_10)))
    ax.set_yticklabels(top_10.index)
    ax.set_xlabel("Total Points", fontsize=12, fontweight='bold')
    ax.set_title("Top 10 Players Comparison (Your player highlighted)", fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    for i, (bar, val) in enumerate(zip(bars, top_10.values)):
        ax.text(val + 1, i, f'{int(val)}', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)

# =====================================================
# 🏆 TEAM ANALYSIS
# =====================================================
elif page == "🏆 Team Analysis":
    st.markdown('<div class="main-header">🏆 Team Analysis</div>', unsafe_allow_html=True)
    
    # Team progression
    st.subheader("📈 Team Progression Through Tournament Rounds")
    
    round_columns = [col for col in matches.columns
                     if any(key in col.upper() for key in ["ROUND", "Q-FINAL", "QF", "SEMI", "FINAL"])]
    
    if round_columns:
        round_points = matches[round_columns].sum(numeric_only=True)
        stage_totals = round_points.reset_index()
        stage_totals.columns = ["Round / Stage", "Total Points"]
        stage_totals["Round / Stage"] = stage_totals["Round / Stage"].astype(str).str.upper().str.strip()
        
        def normalize_stage(s):
            s = s.upper()
            if "Q-FINAL" in s or s == "QF" or "QUARTER" in s:
                return "Q-FINAL"
            if "ROUND 1" in s or s == "R1":
                return "ROUND 1"
            if "ROUND 2" in s or s == "R2":
                return "ROUND 2"
            if "ROUND 3" in s or s == "R3":
                return "ROUND 3"
            if "SEMI" in s:
                return "SEMI-FINAL"
            if "FINAL" in s and "SEMI" not in s:
                return "FINAL"
            return s
        
        stage_totals["Round / Stage"] = stage_totals["Round / Stage"].apply(normalize_stage)
        stage_totals = stage_totals.groupby("Round / Stage", as_index=False)["Total Points"].sum()
        
        desired_order = ['ROUND 1', 'ROUND 2', 'ROUND 3', 'Q-FINAL', 'SEMI-FINAL', 'FINAL']
        stage_totals["Round / Stage"] = pd.Categorical(stage_totals["Round / Stage"],
                                                       categories=desired_order, ordered=True)
        stage_totals = stage_totals.sort_values("Round / Stage").dropna(subset=["Round / Stage"])
        
        nonzero = stage_totals[stage_totals["Total Points"] > 0]
        if not nonzero.empty:
            last_stage = nonzero["Round / Stage"].iloc[-1]
            st.success(f"🏆 *Highest Stage Reached:* {last_stage}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(stage_totals)))
            bars = ax.bar(stage_totals["Round / Stage"].astype(str), stage_totals["Total Points"], 
                         color=colors, edgecolor='black', linewidth=2)
            ax.set_xlabel("Tournament Round", fontsize=12, fontweight='bold')
            ax.set_ylabel("Total Points", fontsize=12, fontweight='bold')
            ax.set_title("Team Performance Across Tournament Stages", fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            
            for bar, val in zip(bars, stage_totals["Total Points"]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(val)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("### 📊 Stage Statistics")
            for idx, row in stage_totals.iterrows():
                st.metric(row["Round / Stage"], f"{int(row['Total Points'])} points")
    
    # Quarter Final Analysis
    st.markdown("---")
    st.subheader("🏅 Quarter Final Performance Analysis")
    
    qfinal_candidates = [col for col in matches.columns 
                        if any(x in col.upper() for x in ["Q-FINAL", "QF", "QUARTER"])]
    
    if qfinal_candidates:
        qcol = qfinal_candidates[0]
        qf_summary = matches.groupby("Player")[qcol].sum().reset_index().sort_values(by=qcol, ascending=False)
        qf_summary = qf_summary[qf_summary[qcol] > 0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(qf_summary.head(10).style.background_gradient(subset=[qcol], cmap='Oranges'), 
                        use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            top_qf = qf_summary.head(10)
            colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_qf)))
            bars = ax.barh(range(len(top_qf)), top_qf[qcol].values, color=colors, edgecolor='black', linewidth=1.5)
            ax.set_yticks(range(len(top_qf)))
            ax.set_yticklabels(top_qf["Player"])
            ax.set_xlabel("Points", fontsize=12, fontweight='bold')
            ax.set_title("Top 10 Players - Quarter Final", fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            
            for i, (bar, val) in enumerate(zip(bars, top_qf[qcol].values)):
                ax.text(val + 0.3, i, f'{int(val)}', va='center', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # Player-wise detailed analysis - REMOVE NaN FROM DROPDOWN
    st.markdown("---")
    st.subheader("🔍 Individual Player Performance in Team Context")
    
    valid_team_players = sorted([p for p in matches["Player"].unique() if p and str(p).upper() != "NAN"])
    
    if not valid_team_players:
        st.error("No valid player data available.")
        st.stop()
    
    selected_team_player = st.selectbox("Select Player for Detailed Analysis", valid_team_players, key="team_player")
    
    if selected_team_player:
        team_df = matches[matches["Player"] == selected_team_player]
        p_total = team_df["Total Points"].sum()
        p_matches = len([t for t in team_df["Tournament"].unique() 
                        if not any(x in str(t).upper() for x in ["TOTAL", "MATCHES"])])
        p_avg = p_total / p_matches if p_matches > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Points", int(p_total))
        col2.metric("Tournaments Played", p_matches)
        col3.metric("Avg Points/Tournament", f"{p_avg:.2f}")
        
        # Performance chart
        tournament_chart = team_df[team_df["Tournament"].str.contains("TOTAL|MATCHES", case=False) == False]
        tournament_chart = tournament_chart.groupby("Tournament")["Total Points"].sum().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(tournament_chart)))
        bars = ax.bar(range(len(tournament_chart)), tournament_chart.values, color=colors, 
                     edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(tournament_chart)))
        ax.set_xticklabels([t[:30] + '...' if len(t) > 30 else t for t in tournament_chart.index], 
                          rotation=45, ha='right')
        ax.set_ylabel("Points", fontsize=12, fontweight='bold')
        ax.set_title(f"{selected_team_player} - Tournament-wise Performance", fontsize=14, fontweight='bold')
        ax.axhline(y=p_avg, color='red', linestyle='--', linewidth=2, label=f'Avg: {p_avg:.1f}')
        ax.legend()
        
        for bar, val in zip(bars, tournament_chart.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(val)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)

# =====================================================
# 📈 ATTENDANCE INSIGHTS
# =====================================================
else:  # Attendance Insights page
    st.markdown('<div class="main-header">📈 Attendance & Performance Insights</div>', unsafe_allow_html=True)
    
    if attendance.empty or "Attendance_pct" not in matches.columns:
        st.warning("⚠ Attendance data not available. Please check the attendance file.")
    else:
        # Attendance overview
        st.subheader("📊 Attendance Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        avg_att = attendance["Attendance_pct"].mean()
        max_att = attendance["Attendance_pct"].max()
        min_att = attendance["Attendance_pct"].min()
        total_players_att = len(attendance)
        
        col1.metric("📈 Average Attendance", f"{avg_att:.1f}%")
        col2.metric("🌟 Highest Attendance", f"{max_att:.1f}%")
        col3.metric("⚠ Lowest Attendance", f"{min_att:.1f}%")
        col4.metric("👥 Players Tracked", total_players_att)
        
        st.markdown("---")
        
        # Attendance distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Attendance Distribution")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            n, bins, patches = ax.hist(attendance["Attendance_pct"], bins=20, 
                                      color='#4ECDC4', alpha=0.7, edgecolor='black')
            
            # Color code bins
            for i, patch in enumerate(patches):
                if bins[i] < 75:
                    patch.set_facecolor('#FF6B6B')
                elif bins[i] < 90:
                    patch.set_facecolor('#FFD93D')
                else:
                    patch.set_facecolor('#6BCF7F')
            
            ax.set_xlabel("Attendance Percentage", fontsize=12, fontweight='bold')
            ax.set_ylabel("Number of Players", fontsize=12, fontweight='bold')
            ax.set_title("Player Attendance Distribution", fontsize=14, fontweight='bold')
            ax.axvline(avg_att, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {avg_att:.1f}%')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("🎯 Attendance Categories")
            
            # Categorize attendance
            def categorize_attendance(pct):
                if pct >= 90:
                    return "Excellent (≥90%)"
                elif pct >= 75:
                    return "Good (75-89%)"
                elif pct >= 60:
                    return "Average (60-74%)"
                else:
                    return "Poor (<60%)"
            
            attendance["Category"] = attendance["Attendance_pct"].apply(categorize_attendance)
            cat_counts = attendance["Category"].value_counts()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors_cat = ['#6BCF7F', '#FFD93D', '#FFA07A', '#FF6B6B']
            wedges, texts, autotexts = ax.pie(cat_counts.values, labels=cat_counts.index, 
                                               autopct='%1.1f%%', startangle=90,
                                               colors=colors_cat, textprops={'fontsize': 11, 'fontweight': 'bold'})
            ax.set_title("Attendance Category Distribution", fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Top and bottom attendance
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌟 Top 10 Attendance")
            top_att = attendance.nlargest(10, "Attendance_pct")[["Player", "Attendance_pct", "Present", "Total"]]
            st.dataframe(top_att.style.background_gradient(subset=["Attendance_pct"], cmap='Greens'),
                        use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("⚠ Bottom 10 Attendance")
            bottom_att = attendance.nsmallest(10, "Attendance_pct")[["Player", "Attendance_pct", "Present", "Total"]]
            st.dataframe(bottom_att.style.background_gradient(subset=["Attendance_pct"], cmap='Reds'),
                        use_container_width=True, hide_index=True)
        
        # Performance vs Attendance Analysis - CLEANED VERSION (NO SIDEBAR)
        st.markdown("---")
        st.subheader("🔬 Performance vs Attendance Correlation Analysis")
        
        # Create comprehensive performance-attendance dataset
        perf_att = matches.groupby("Player").agg({
            "Total Points": "sum",
            "Tournament": "nunique",
            "Attendance_pct": "first"
        }).reset_index()
        
        # Calculate average points per tournament for fairer comparison
        perf_att["Avg Points per Tournament"] = perf_att["Total Points"] / perf_att["Tournament"]
        perf_att = perf_att.dropna(subset=["Attendance_pct"])
        
        if not perf_att.empty and len(perf_att) > 1:
            # Create comprehensive scatter plot - FULL WIDTH
            fig = plt.figure(figsize=(18, 8))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            
            # Main scatter plot
            ax_main = fig.add_subplot(gs[:, 0])
            
            # Calculate quartiles for better categorization
            att_median = perf_att["Attendance_pct"].median()
            perf_median = perf_att["Avg Points per Tournament"].median()
            
            # Color code by quadrant
            colors = []
            for _, row in perf_att.iterrows():
                if row["Attendance_pct"] >= att_median and row["Avg Points per Tournament"] >= perf_median:
                    colors.append('#2ecc71')  # Green - Star players
                elif row["Attendance_pct"] >= att_median and row["Avg Points per Tournament"] < perf_median:
                    colors.append('#f39c12')  # Orange - Potential
                elif row["Attendance_pct"] < att_median and row["Avg Points per Tournament"] >= perf_median:
                    colors.append('#3498db')  # Blue - Inconsistent stars
                else:
                    colors.append('#e74c3c')  # Red - Needs attention
            
            # Create scatter plot
            scatter = ax_main.scatter(perf_att["Attendance_pct"], 
                                    perf_att["Avg Points per Tournament"],
                                    s=perf_att["Tournament"] * 80,
                                    c=colors,
                                    alpha=0.6,
                                    edgecolors='black',
                                    linewidth=2)
            
            # Add quadrant lines
            ax_main.axhline(y=perf_median, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
            ax_main.axvline(x=att_median, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
            
            # Add trend line
            z = np.polyfit(perf_att["Attendance_pct"], perf_att["Avg Points per Tournament"], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(perf_att["Attendance_pct"].min(), perf_att["Attendance_pct"].max(), 100)
            ax_main.plot(x_trend, p(x_trend), "r-", linewidth=3, alpha=0.8, label='Trend Line')
            
            # Label top performers
            top_5 = perf_att.nlargest(5, "Avg Points per Tournament")
            for _, row in top_5.iterrows():
                ax_main.annotate(row["Player"], 
                               (row["Attendance_pct"], row["Avg Points per Tournament"]),
                               xytext=(8, 8), 
                               textcoords="offset points", 
                               fontsize=9,
                               fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7, edgecolor='black'),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='black', lw=1.5))
            
            ax_main.set_xlabel("Attendance (%)", fontsize=13, fontweight='bold')
            ax_main.set_ylabel("Average Points per Tournament", fontsize=13, fontweight='bold')
            ax_main.set_title("Performance vs Attendance: Comprehensive Analysis", fontsize=15, fontweight='bold')
            ax_main.grid(alpha=0.3, linestyle='--')
            ax_main.legend(fontsize=11)
            
            # Add quadrant labels
            y_range = perf_att["Avg Points per Tournament"].max() - perf_att["Avg Points per Tournament"].min()
            x_range = perf_att["Attendance_pct"].max() - perf_att["Attendance_pct"].min()
            
            ax_main.text(perf_att["Attendance_pct"].max() - x_range*0.05, 
                       perf_att["Avg Points per Tournament"].max() - y_range*0.05,
                       '⭐ STAR PLAYERS', 
                       ha='right', va='top', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.3, edgecolor='black', linewidth=2))
            
            ax_main.text(perf_att["Attendance_pct"].min() + x_range*0.05, 
                       perf_att["Avg Points per Tournament"].max() - y_range*0.05,
                       '⚡ TALENT\n(Low Att)', 
                       ha='left', va='top', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='#3498db', alpha=0.3, edgecolor='black', linewidth=2))
            
            ax_main.text(perf_att["Attendance_pct"].max() - x_range*0.05, 
                       perf_att["Avg Points per Tournament"].min() + y_range*0.05,
                       '💎 POTENTIAL\n(High Att)', 
                       ha='right', va='bottom', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='#f39c12', alpha=0.3, edgecolor='black', linewidth=2))
            
            ax_main.text(perf_att["Attendance_pct"].min() + x_range*0.05, 
                       perf_att["Avg Points per Tournament"].min() + y_range*0.05,
                       '⚠ ATTENTION\nNEEDED', 
                       ha='left', va='bottom', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.3, edgecolor='black', linewidth=2))
            
            # Top right: Distribution of attendance
            ax_att = fig.add_subplot(gs[0, 1])
            ax_att.hist(perf_att["Attendance_pct"], bins=15, color='#3498db', alpha=0.7, edgecolor='black')
            ax_att.axvline(att_median, color='red', linestyle='--', linewidth=2, label=f'Median: {att_median:.1f}%')
            ax_att.set_xlabel("Attendance %", fontsize=10, fontweight='bold')
            ax_att.set_ylabel("Count", fontsize=10, fontweight='bold')
            ax_att.set_title("Attendance Distribution", fontsize=11, fontweight='bold')
            ax_att.legend(fontsize=8)
            ax_att.grid(alpha=0.3)
            
            # Bottom right: Distribution of performance
            ax_perf = fig.add_subplot(gs[1, 1])
            ax_perf.hist(perf_att["Avg Points per Tournament"], bins=15, color='#2ecc71', alpha=0.7, edgecolor='black')
            ax_perf.axvline(perf_median, color='red', linestyle='--', linewidth=2, label=f'Median: {perf_median:.1f}')
            ax_perf.set_xlabel("Avg Points/Tournament", fontsize=10, fontweight='bold')
            ax_perf.set_ylabel("Count", fontsize=10, fontweight='bold')
            ax_perf.set_title("Performance Distribution", fontsize=11, fontweight='bold')
            ax_perf.legend(fontsize=8)
            ax_perf.grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.caption("💡 *Bubble size* = Number of tournaments played | *Colors*: Green=Stars, Orange=Potential, Blue=Inconsistent, Red=Needs Attention")
        
        # Detailed player analysis with attendance
        st.markdown("---")
        st.subheader("🔍 Player-wise Detailed Analysis")
        
        player_analysis = perf_att.copy()
        
        # Add rankings
        player_analysis["Performance Rank"] = player_analysis["Avg Points per Tournament"].rank(ascending=False, method='min').astype(int)
        player_analysis["Attendance Rank"] = player_analysis["Attendance_pct"].rank(ascending=False, method='min').astype(int)
        
        # Add categories
        player_analysis["Performance Level"] = player_analysis["Avg Points per Tournament"].apply(
            lambda x: "🌟 Excellent" if x >= player_analysis["Avg Points per Tournament"].quantile(0.75)
            else "✅ Good" if x >= player_analysis["Avg Points per Tournament"].quantile(0.5)
            else "📊 Average" if x >= player_analysis["Avg Points per Tournament"].quantile(0.25)
            else "⚠ Below Average"
        )
        
        def categorize_attendance(pct):
            if pct >= 90:
                return "Excellent (≥90%)"
            elif pct >= 75:
                return "Good (75-89%)"
            elif pct >= 60:
                return "Average (60-74%)"
            else:
                return "Poor (<60%)"
        
        player_analysis["Attendance Level"] = player_analysis["Attendance_pct"].apply(categorize_attendance)
        
        # Add efficiency score (weighted combination)
        player_analysis["Efficiency Score"] = (
            player_analysis["Avg Points per Tournament"] / player_analysis["Avg Points per Tournament"].max() * 0.6 +
            player_analysis["Attendance_pct"] / 100 * 0.4
        ) * 100
        
        player_analysis = player_analysis.sort_values("Efficiency Score", ascending=False)
        
        # Display sortable dataframe
        st.dataframe(
            player_analysis[["Player", "Total Points", "Tournament", "Avg Points per Tournament", 
                           "Performance Rank", "Attendance_pct", "Attendance Rank", 
                           "Efficiency Score", "Performance Level", "Attendance Level"]]
            .style.background_gradient(subset=["Avg Points per Tournament"], cmap='Greens')
            .background_gradient(subset=["Attendance_pct"], cmap='Blues')
            .background_gradient(subset=["Efficiency Score"], cmap='RdYlGn'),
            use_container_width=True,
            hide_index=True
        )
        
        st.caption("💡 *Efficiency Score* = Weighted combination of performance (60%) and attendance (40%)")
        
        # Radar chart for top 5 players
        st.markdown("---")
        st.subheader("🎯 Top 5 Players: Multi-Dimensional Analysis")
        
        top_5_players = player_analysis.head(5)
        
        fig, axes = plt.subplots(1, 5, figsize=(20, 4), subplot_kw=dict(projection='polar'))
        
        categories = ['Performance', 'Attendance', 'Consistency', 'Participation']
        
        for idx, (ax, (_, player_row)) in enumerate(zip(axes, top_5_players.iterrows())):
            # Normalize values to 0-100 scale
            values = [
                (player_row["Avg Points per Tournament"] / player_analysis["Avg Points per Tournament"].max()) * 100,
                player_row["Attendance_pct"],
                100 - ((player_row["Avg Points per Tournament"] / player_analysis["Avg Points per Tournament"].std()) if player_analysis["Avg Points per Tournament"].std() > 0 else 50),
                (player_row["Tournament"] / player_analysis["Tournament"].max()) * 100
            ]
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]
            angles += angles[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=player_row["Player"], color=plt.cm.Set2(idx))
            ax.fill(angles, values, alpha=0.25, color=plt.cm.Set2(idx))
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=8)
            ax.set_ylim(0, 100)
            ax.set_title(player_row["Player"], fontsize=11, fontweight='bold', pad=20)
            ax.grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.caption("💡 Radar charts show normalized scores across 4 dimensions: Performance, Attendance, Consistency, and Participation")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 2rem; background-color: #f8f9fa; border-radius: 10px;'>
        <h3 style='color: #FF6B35;'>🏐 Kabaddi Analytics Dashboard</h3>
        <p style='color: #6c757d;'>Developed for College Sports Analytics | Data-Driven Team Performance Insights</p>
    </div>
""", unsafe_allow_html=True)