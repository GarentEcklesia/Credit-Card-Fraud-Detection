import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model pipeline"""
    try:
        model_package = joblib.load('fraud_detection_model.joblib')
        return model_package
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model_package = load_model()

st.title("🏦 Credit Card Fraud Detection System")
st.markdown("### AI-Powered Real-Time Transaction Analysis")
st.markdown("---")

if model_package is None:
    st.error("❌ Failed to load the model. Please ensure 'fraud_detection_model.joblib' is in the same directory.")
    st.stop()

with st.sidebar:
    st.header("📊 Model Information")
    st.info(f"**Model:** {model_package['model_name']}")
    st.info(f"**Trained:** {model_package['training_date']}")
    
    st.markdown("### 🎯 Performance Metrics")
    metrics = model_package['performance_metrics']
    st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    st.metric("Precision", f"{metrics['precision']:.2%}")
    st.metric("Recall", f"{metrics['recall']:.2%}")
    st.metric("F1-Score", f"{metrics['f1_score']:.2%}")
    st.metric("ROC-AUC", f"{metrics['roc_auc']:.2%}")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This system uses machine learning to detect 
    fraudulent credit card transactions in real-time.
    
    **Features:**
    - Real-time prediction
    - Probability scores
    - Risk assessment
    - Feature importance
    """)

tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📊 Batch Prediction", "📈 Model Insights"])

with tab1:
    st.header("Single Transaction Analysis")
    st.markdown("Enter transaction details below to check for potential fraud.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transaction Details")
        distance_from_home = st.number_input(
            "Distance from Home (km)",
            min_value=0.0,
            max_value=10000.0,
            value=10.0,
            step=0.1,
            help="How far from home did this transaction occur?"
        )
        
        distance_from_last = st.number_input(
            "Distance from Last Transaction (km)",
            min_value=0.0,
            max_value=10000.0,
            value=5.0,
            step=0.1,
            help="Distance from the previous transaction location"
        )
        
        ratio_to_median = st.number_input(
            "Ratio to Median Purchase Price",
            min_value=0.0,
            max_value=100.0,
            value=1.0,
            step=0.1,
            help="Transaction amount compared to your typical spending"
        )
    
    with col2:
        st.subheader("Transaction Method")
        repeat_retailer = st.selectbox(
            "Repeat Retailer?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Have you purchased from this retailer before?"
        )
        
        used_chip = st.selectbox(
            "Used Chip?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Was the chip reader used?"
        )
        
        used_pin = st.selectbox(
            "Used PIN?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Was a PIN number entered?"
        )
        
        online_order = st.selectbox(
            "Online Order?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Was this an online transaction?"
        )
    
    if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
        input_data = pd.DataFrame({
            'distance_from_home': [distance_from_home],
            'distance_from_last_transaction': [distance_from_last],
            'ratio_to_median_purchase_price': [ratio_to_median],
            'repeat_retailer': [repeat_retailer],
            'used_chip': [used_chip],
            'used_pin_number': [used_pin],
            'online_order': [online_order]
        })
        
        prediction = model_package['pipeline'].predict(input_data)[0]
        probability = model_package['pipeline'].predict_proba(input_data)[0]

        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.error("### ⚠️ FRAUD DETECTED")
                st.markdown(f"**Fraud Probability:** {probability[1]:.1%}")
            else:
                st.success("### ✅ LEGITIMATE")
                st.markdown(f"**Legitimate Probability:** {probability[0]:.1%}")
        
        with col2:
            fraud_prob = probability[1]
            if fraud_prob < 0.3:
                risk = "🟢 LOW RISK"
                risk_color = "green"
            elif fraud_prob < 0.7:
                risk = "🟡 MEDIUM RISK"
                risk_color = "orange"
            else:
                risk = "🔴 HIGH RISK"
                risk_color = "red"
            
            st.markdown(f"### {risk}")
            st.markdown(f"**Risk Score:** {fraud_prob:.1%}")
        
        with col3:
            if prediction == 1:
                st.markdown("### 💡 Recommendation")
                st.markdown("**Action:** Block transaction and contact customer")
            else:
                st.markdown("### 💡 Recommendation")
                st.markdown("**Action:** Approve transaction")

        st.markdown("---")
        st.subheader("📊 Fraud Probability Gauge")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=probability[1] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fraud Probability (%)"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if probability[1] > 0.5 else "darkgreen"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "lightyellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 Transaction Details Summary"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                - **Distance from Home:** {distance_from_home} km
                - **Distance from Last Transaction:** {distance_from_last} km
                - **Ratio to Median Price:** {ratio_to_median}
                """)
            with col2:
                st.markdown(f"""
                - **Repeat Retailer:** {'Yes' if repeat_retailer else 'No'}
                - **Used Chip:** {'Yes' if used_chip else 'No'}
                - **Used PIN:** {'Yes' if used_pin else 'No'}
                - **Online Order:** {'Yes' if online_order else 'No'}
                """)

with tab2:
    st.header("Batch Transaction Analysis")
    st.markdown("Upload a CSV file with multiple transactions for batch analysis.")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with the required columns"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! {len(df)} transactions loaded.")

            with st.expander("👀 Preview Data"):
                st.dataframe(df.head(10))

            required_columns = model_package['feature_names']
            missing_columns = set(required_columns) - set(df.columns)
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {missing_columns}")
            else:
                if st.button("🚀 Analyze All Transactions", type="primary"):
                    with st.spinner("Analyzing transactions..."):
                        predictions = model_package['pipeline'].predict(df[required_columns])
                        probabilities = model_package['pipeline'].predict_proba(df[required_columns])

                        df['Prediction'] = predictions
                        df['Fraud_Probability'] = probabilities[:, 1]
                        df['Risk_Level'] = pd.cut(
                            df['Fraud_Probability'],
                            bins=[0, 0.3, 0.7, 1.0],
                            labels=['Low', 'Medium', 'High']
                        )

                    st.markdown("---")
                    st.subheader("📊 Analysis Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total_transactions = len(df)
                    fraud_detected = (predictions == 1).sum()
                    legitimate = (predictions == 0).sum()
                    avg_fraud_prob = df['Fraud_Probability'].mean()
                    
                    col1.metric("Total Transactions", total_transactions)
                    col2.metric("Fraud Detected", fraud_detected, delta=f"{fraud_detected/total_transactions:.1%}")
                    col3.metric("Legitimate", legitimate, delta=f"{legitimate/total_transactions:.1%}")
                    col4.metric("Avg Fraud Probability", f"{avg_fraud_prob:.1%}")

                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_pie = px.pie(
                            values=[fraud_detected, legitimate],
                            names=['Fraud', 'Legitimate'],
                            title='Transaction Distribution',
                            color_discrete_sequence=['#ff6b6b', '#51cf66']
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        risk_counts = df['Risk_Level'].value_counts()
                        fig_bar = px.bar(
                            x=risk_counts.index,
                            y=risk_counts.values,
                            title='Risk Level Distribution',
                            labels={'x': 'Risk Level', 'y': 'Count'},
                            color=risk_counts.index,
                            color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)

                    st.markdown("---")
                    st.subheader("📋 Detailed Results")
      
                    filter_option = st.selectbox(
                        "Filter by:",
                        ["All Transactions", "Fraud Only", "Legitimate Only", "High Risk", "Medium Risk", "Low Risk"]
                    )
                    
                    if filter_option == "Fraud Only":
                        filtered_df = df[df['Prediction'] == 1]
                    elif filter_option == "Legitimate Only":
                        filtered_df = df[df['Prediction'] == 0]
                    elif filter_option == "High Risk":
                        filtered_df = df[df['Risk_Level'] == 'High']
                    elif filter_option == "Medium Risk":
                        filtered_df = df[df['Risk_Level'] == 'Medium']
                    elif filter_option == "Low Risk":
                        filtered_df = df[df['Risk_Level'] == 'Low']
                    else:
                        filtered_df = df
  
                    st.dataframe(
                        filtered_df.style.background_gradient(
                            subset=['Fraud_Probability'],
                            cmap='RdYlGn_r'
                        ),
                        use_container_width=True,
                        height=400
                    )
            
                    csv = filtered_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"fraud_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    else:
        st.info("👆 Please upload a CSV file to begin batch analysis.")

        with st.expander("ℹ️ Expected CSV Format"):
            st.markdown("""
            Your CSV file should contain the following columns:
            
            1. `distance_from_home` - Distance from home (numeric)
            2. `distance_from_last_transaction` - Distance from last transaction (numeric)
            3. `ratio_to_median_purchase_price` - Ratio to median price (numeric)
            4. `repeat_retailer` - Repeat retailer (0 or 1)
            5. `used_chip` - Used chip (0 or 1)
            6. `used_pin_number` - Used PIN (0 or 1)
            7. `online_order` - Online order (0 or 1)
            """)
     
            sample_data = pd.DataFrame({
                'distance_from_home': [10.5, 50.2, 150.0],
                'distance_from_last_transaction': [5.0, 20.0, 100.0],
                'ratio_to_median_purchase_price': [1.0, 2.5, 5.0],
                'repeat_retailer': [1, 0, 1],
                'used_chip': [1, 1, 0],
                'used_pin_number': [1, 0, 0],
                'online_order': [0, 1, 1]
            })
            
            st.dataframe(sample_data)

with tab3:
    st.header("Model Insights & Performance")
   
    st.subheader("📊 Model Performance Metrics")
    metrics = model_package['performance_metrics']
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    col2.metric("Precision", f"{metrics['precision']:.2%}")
    col3.metric("Recall", f"{metrics['recall']:.2%}")
    col4.metric("F1-Score", f"{metrics['f1_score']:.2%}")
    col5.metric("ROC-AUC", f"{metrics['roc_auc']:.2%}")
  
    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown("""
        - **Accuracy:** Overall correctness of predictions
        - **Precision:** Of all predicted frauds, how many were actually fraud
        - **Recall:** Of all actual frauds, how many did we catch
        - **F1-Score:** Balance between Precision and Recall
        - **ROC-AUC:** Model's ability to distinguish between classes
        """)

    st.markdown("---")
    st.subheader("⚙️ Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model Type:**")
        st.info(model_package['model_name'])
        
        st.markdown("**Training Date:**")
        st.info(model_package['training_date'])
    
    with col2:
        st.markdown("**Best Hyperparameters:**")
        params_df = pd.DataFrame(
            list(model_package['best_params'].items()),
            columns=['Parameter', 'Value']
        )
        st.dataframe(params_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🎯 Feature Importance")
    
    try:
        classifier = model_package['pipeline'].named_steps['classifier']
        
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            
            feature_engineer = model_package['pipeline'].named_steps['feature_engineer']
            sample_data = pd.DataFrame([{col: 0 for col in model_package['feature_names']}])
            engineered_features = feature_engineer.transform(sample_data)
            feature_names = engineered_features.columns.tolist()
            
            importance_df = pd.DataFrame({
                'Feature': feature_names[:len(importances)],
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(
                importance_df.head(15),
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 15 Most Important Features',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
        elif hasattr(classifier, 'coef_'):
            coefficients = abs(classifier.coef_[0])
            
            coef_df = pd.DataFrame({
                'Feature': model_package['feature_names'],
                'Coefficient': coefficients
            }).sort_values('Coefficient', ascending=False)
            
            fig = px.bar(
                coef_df.head(15),
                x='Coefficient',
                y='Feature',
                orientation='h',
                title='Top 15 Most Important Features (Absolute Coefficients)',
                color='Coefficient',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")
    
    except Exception as e:
        st.warning(f"Could not extract feature importance: {str(e)}")
    
    st.markdown("---")
    st.subheader("💡 Key Insights")
    
    st.markdown("""
    **What makes a transaction suspicious?**
    
    Based on our model analysis, fraudulent transactions typically show:
    
    1. **Unusual Location Patterns**
       - Large distances from home
       - Multiple transactions from different locations in short time
    
    2. **Abnormal Purchase Behavior**
       - Much higher amounts than usual spending patterns
       - Purchases from unfamiliar retailers
    
    3. **Security Compromises**
       - Transactions without chip or PIN
       - Online orders with unusual patterns
    
    **Recommendations for Users:**
    - Enable transaction notifications
    - Use chip and PIN when possible
    - Monitor unusual location patterns
    - Report suspicious activity immediately
    """)

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p>🏦 Credit Card Fraud Detection System | Built with Streamlit & Machine Learning</p>
        <p>⚠️ For demonstration purposes only. Always consult with financial security professionals.</p>
    </div>
    """,
    unsafe_allow_html=True
)