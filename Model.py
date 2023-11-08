import streamlit as st
import pandas as pd
from src.pipeline.predict import PredictPipeline

def get_space(k):
    for i in range(k):
        st.write("")
        
def main():
    st.set_page_config(
    page_title="Loan Approval Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
    )
    
    with st.container():
        st.title("Loan Approval Predictor") 
        get_space(3)

        col1,_=st.columns([1.25,1.75])

        with col1:
            
            data = pd.read_csv("data/loan_approval.csv")
            data=data.iloc[:,2:12]
            input_dict, features = {},{}

            data= data.drop(['Exited'], axis=1)
            df=data.copy()
            df.columns=['Credit_Score', 'Gender', 'Age', 'Tenure',
                'Balance', 'Num_Of_Products', 'Credit_Card', 'Is_Active_Member',
                'Salary']
            col=dict(zip(data.columns, df.columns))
            
            with st.form("my_form",clear_on_submit=True):
                for i in data.columns:
                    if data[i].dtype=='int64':
                        if (len(data[i].value_counts()))==2:
                            val=st.radio(col[i],['yes','no'],index=None)
                            input_dict[i]=1 if val == 'yes' else 0
                        else :
                            input_dict[i]=st.number_input(col[i], value=None, step=1)
                    if data[i].dtype=='float64':
                        input_dict[i]=st.number_input(col[i], value=None, step=10000.0)
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
                st.error("Loan couldn't be approved")
            else :
                st.success("Loan could be approved")
            
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