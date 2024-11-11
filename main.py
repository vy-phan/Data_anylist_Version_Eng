import streamlit as st
import pandas as pd
import google.generativeai as genai
import os 

# Đặt page config ngay đầu file, trước khi gọi bất kỳ st command nào khác
st.set_page_config(
    page_title="Assistant for Data Analysis",
    page_icon="🔍",
    layout="centered"
)

data = st.file_uploader(":file_folder: Upload a file",type=(["csv","tsv","xlsx","xls"]))

def init_session_state():
    if "API_KEY" not in st.session_state:
        st.session_state.API_KEY = ""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "prompt_value" not in st.session_state:
        st.session_state.prompt_value = ""
# function chart 
def create_line_chart(df, x_col , y_col):
    if x_col == None or y_col == None:
        if x_col == None : 
            st.line_chart(df[y_col])
        if y_col == None :
            st.line_chart(df[x_col])
    else:
        st.line_chart(df.set_index(x_col)[y_col])

def create_bar_chart(df, x_col , y_col):
    if x_col == None or y_col == None:
        if x_col == None : 
            st.bar_chart(df[y_col])
        if y_col == None :
            st.bar_chart(df[x_col])
    else:
        st.bar_chart(df.set_index(x_col)[y_col])

def create_scatter_chart(df, x_col , y_col):
    st.scatter_chart(df.set_index(x_col)[y_col])

def create_area_chart(df, x_col , y_col):
    if x_col == None or y_col == None:
        if x_col == None : 
            st.area_chart(df[y_col])
        if y_col == None :
            st.area_chart(df[x_col])
    else:
        st.area_chart(df.set_index(x_col)[y_col])

# function create new data 
def create_new_columndf(df,col1_new_i, col2_new_i,col_operation, col_new_name):
    operation = ''
    if col_operation == " Add ( + ) ":
        operation = "+"
    elif col_operation == " Multi ( * ) ":
        operation = "*" 
    elif col_operation == " Subtract ( - ) ":
        operation = "-"     
    elif col_operation == " Divide ( / ) ":
        operation = "/"
    else:
        operation = "Lựa chọn không hợp lệ"
        
    st.sidebar.write(col1_new_i + operation + col2_new_i + "=" + col_new_name)     
    df[col1_new_i] = pd.to_numeric(df[col1_new_i], errors='coerce')
    df[col2_new_i] = pd.to_numeric(df[col2_new_i], errors='coerce')

    if operation != '':
        if operation == "+":
            df[col_new_name] = df[col1_new_i] + df[col2_new_i]
        if operation == "*":
            df[col_new_name] = df[col1_new_i] * df[col2_new_i]
        if operation == "-":
            df[col_new_name] = df[col1_new_i] - df[col2_new_i]
        if operation == "/":
            df[col_new_name] = df[col1_new_i] / df[col2_new_i]
    return df

def create_prompt_template(prompt: str, rows: list) -> str:
    """Create standard template for prompt"""
    template = f"""
    You will act as a data analysis expert and answer all questions raised by users.
    Question: {prompt}
    Below are two rows from the CSV file converted to dictionary format:
    {rows}
    Response format requirements:
    1. Answer questions related to the data
    2. Detailed analysis and provide recommendations related to the data
    3. Specific examples (if any) 
    4. Conclusion
    
    Please respond with well-formatted text.
    """
    return template

def create_analysis(data) -> str:
    return f""""
     Below is a row from the CSV file converted to dictionary format:
     {data}
    Please follow these instructions: \n\n

    1. **Analyze data and suggest appropriate chart types for the columns.**
    When making suggestions, please only provide the chart name and corresponding columns clearly and correctly.
    Column names must be in uppercase. And minimum 1 column, maximum 2 columns for each chart type.
    **Note:** Only suggest the following chart types: line chart, bar chart, scatter chart, and area chart.
    If there is no suitable chart for a column, skip it.
    
    2. **Suggest new column names** by combining existing columns using only addition, subtraction, multiplication, and division operations.
    Just output the new column name and the mathematical expression (e.g., 'Total Sales' = 'Sales' + 'Discount').

    3. **Empty response:** If there are any issues with the data or if you cannot find suitable charts for any columns, leave the response empty.
    """

def clear_prompt():
    st.session_state.prompt_value = ""

def create_chat(rows):
    if "API_KEY" in st.session_state and st.session_state.API_KEY:
        # Input cho prompt
        if "prompt_value" not in st.session_state:
            st.session_state.prompt_value = ""

        prompt = st.text_input(
            "Enter your prompt", 
            key="prompt_input",
            value=st.session_state.prompt_value,
            on_change=clear_prompt
        )

        col1, col2 = st.columns(2)
        with col1:
            send_button = st.button("Send", key="send_button")
        with col2:
            send_analysis_button = st.button("Send Analysis", key="send_analysis_button")

        # Xử lý khi người dùng gửi prompt
        if send_button and prompt:
            # Gửi prompt và nhận response
            prompts = create_prompt_template(prompt, rows)
            response = st.session_state.chat.send_message(prompts)
            
            # Lưu vào messages
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            
            clear_prompt()

        if send_analysis_button:
            analysis = create_analysis(rows)
            response = st.session_state.chat.send_message(analysis)
            st.write(response.text)


        # Hiển thị chat history từ messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.write("🧑 You:", message["content"])
            else:
                st.write("🤖 Assistant:", message["content"])

def init_chat(df,rows):
        #I.) preview data
        st.subheader("Data Preview")
        st.info("Preview data with first 5 rows")
        st.write(df.head())

        # describe data
        st.subheader("Data Summary")
        st.info("Basic general information of the dataf")
        st.write(df.describe())

        #II.) filter data
        st.subheader("Filter Data")
        st.info("Select detail each values in columns")
        columnes = df.columns.tolist()
        col_1 , col_2 = st.columns(2)
        with col_1:
            selected_col = st.selectbox("Select columns to filter by",columnes)
        unique_val = df[selected_col].unique()
        with col_2:    
            selected_val = st.selectbox("Select value",unique_val)

        filter_df = df[df[selected_col]== selected_val] 
        st.write(filter_df)
        
        
        # option chart 
        Chart_Name = ['line_chart','bar_chart','scatter_chart','area_chart']
        
        # create new column in NAVBAR
        col1_new_i = st.sidebar.selectbox("Select col 1",columnes)
        col2_new_i = st.sidebar.selectbox("Select col 2",columnes)
        col_operation = st.sidebar.selectbox("Select caculation",[" Add ( + ) "," Multi ( * ) "," Subtract ( - ) "," Divide ( / ) "])
        col_new_name = st.sidebar.text_input("Col name")

        # Khởi tạo flag trong session state
        if 'flag' not in st.session_state:
            st.session_state.flag = False
            
        # Khởi tạo modified_df trong session state để lưu DataFrame sau khi thay đổi
        if 'modified_df' not in st.session_state:
            st.session_state.modified_df = df.copy()

        # Xử lý tạo cột mới
        if col1_new_i and col2_new_i and col_operation and col_new_name:
            st.sidebar.info(f"You are make sure chose column name with {col_new_name} ?")
            if st.sidebar.button("Create Column"):
                st.session_state.flag = True
                # Tạo bản sao của DataFrame trước khi thay đổi
                temp_df = st.session_state.modified_df.copy()
                # Tạo cột mới
                temp_df = create_new_columndf(temp_df, col1_new_i, col2_new_i, col_operation, col_new_name)
                # Reset index để đảm bảo tính liên tục
                st.session_state.modified_df = temp_df.reset_index(drop=True)

        st.subheader("Plot data")
        # Sử dụng DataFrame phù hợp
        current_df = st.session_state.modified_df if st.session_state.flag else df
        # Lấy danh sách cột
        columns_after = current_df.columns.tolist()
        columns_after.insert(0, None)

        #a) select columns 
        st.info("Generate chart with [ column x ] , [ column y ]")
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select x-axis col", columns_after)
        with col2:
            y_col = st.selectbox("Select y-axis col", columns_after)

        Chart_Val = st.selectbox("Enter your chart you want", Chart_Name) 

        #b) show data after   
        st.write("Preview of current data:")
        st.write(current_df.head())

        # Thêm thông tin về index
        st.info(f"Total rows: {len(current_df)}, Index range: 0 to {len(current_df)-1}")

        #c) button create chart
        if st.button("Create Chart"):
            if Chart_Val == 'line_chart':
                create_line_chart(current_df, x_col, y_col)
            elif Chart_Val == 'bar_chart':
                create_bar_chart(current_df, x_col, y_col)
            elif Chart_Val == 'scatter_chart':
                create_scatter_chart(current_df, x_col, y_col)    
            elif Chart_Val == 'area_chart':
                create_area_chart(current_df, x_col, y_col)

        st.markdown("### Please enter your API key if you want to use chatbot")
        API_KEY = st.text_input("Enter your API key", type="password", key="api_input")
        button = st.button("Save API key", key="save_button")

        if button:
            if API_KEY.startswith("AIzaSy"):
                st.session_state.API_KEY = API_KEY
                genai.configure(api_key=st.session_state.API_KEY)
                st.session_state.model = genai.GenerativeModel("gemini-1.5-flash")
                st.session_state.chat = st.session_state.model.start_chat()
                st.success("API key saved successfully!")
                
            else:
                st.error("Invalid API key")

        init_session_state()
            
        create_chat(rows)

def main():
    if data is not None:
        filename = data.name
        st.success(filename)
        df = pd.read_csv(data)
            # append into rows
        rows = []
        for i in range(3):
            rows.append(df.iloc[i].to_dict())

        init_chat(df,rows)




if __name__ == "__main__":
    main()
