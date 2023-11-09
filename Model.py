import streamlit as st
import pandas as pd
from src.pipeline.predict import PredictPipeline

def get_space(k):
    for i in range(k):
        st.write("")
        
def main():
    st.set_page_config(
    page_title="Customer Churn Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
    )
    
    with st.container():
        st.title("Customer Churn Predictor")        
        get_space(3)

        col1,_=st.columns([0.75,2.25])

        with col1:
            
            data = pd.read_csv("data/churn.csv")
            data=data.iloc[:,2:12]
            input_dict, features = {},{}

            data= data.drop(['Exited'], axis=1)
            df=data.copy()
            df.columns=['Credit_Score', 'Gender', 'Age', 'Tenure ( in years )',
                'Balance', 'Num_Of_Products', 'Credit_Card', 'Is_Active_Member',
                'Salary (Monthly)']
            col=dict(zip(data.columns, df.columns))
            max_val={'CreditScore':950, 'Age':95 , 'Tenure':50,
                'Balance':1000000.00, 'NumOfProducts':4,
                'EstimatedSalary':200000.00}

            with st.form("my_form",clear_on_submit=True):
                for i in data.columns:
                    if data[i].dtype=='int64':
                        if (len(data[i].value_counts()))==2:
                            val=st.radio(col[i],['yes','no'],index=None)
                            input_dict[i]=1 if val == 'yes' else 0
                        else :
                            input_dict[i]=st.slider(col[i], 0, max_val[i])
                    if data[i].dtype=='float64':
                        input_dict[i]=st.slider(col[i],  0.0, max_val[i])
                    if data[i].dtype=='object':
                        input_dict[i]=st.radio(col[i],list(data[i].value_counts().keys()),index=None)
                        
                submitted = st.form_submit_button("Submit")
                if submitted:
                    if "" in list(input_dict.values()) or None in list(input_dict.values()):
                        st.error("Enter all details in order to submit form")
                    else:
                        features=input_dict
        
        get_space(3)
        if features:
            print(features)
            predict_pipeline=PredictPipeline()
            result=predict_pipeline.predict(pd.DataFrame(features,index=[0]))
            print(result[0])
            
            if result[0][0]>0.5:
                st.error("Customer may leave the bank")
            else :
                st.success("Customer could stay back")
            
        else:
            st.warning("To obtain the result,  fill out the form completely")
            
        get_space(5)
            
        '''
        #### Limitations
        * **This app can assist with the approval process, but it should not be used as a substitute of professional approvals**
        * **Providing illogical or uncertain data may lead to false results**
        '''

if __name__ == '__main__':
  main()