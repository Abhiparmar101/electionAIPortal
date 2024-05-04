from flask import Flask, request, jsonify,session
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import os
import pandas as pd
from flask_session import Session
from flask_cors import CORS
import sqlite3
import base64
from io import BytesIO
import uuid
import logging
app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
CORS(app, origins=["*"])
os.environ["OPENAI_API_KEY"] = ""
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def fetch_full_outer_join_data():
    # """Simulate a FULL OUTER JOIN and fetch data from SQLite."""
    # conn = sqlite3.connect('metadata.db')
    # # First LEFT JOIN: metadata LEFT JOIN camera_installation_details
    # query_left = """
    # SELECT m.id, m.cameraid as cameraid, m.timestamp, m.imageurl, m.personcount,
    #        c.SR, c.AC_Name, c.Part_No_Polling_Station_Name, c.Technician_Contact_Number, c.Technican_Name, c.zone
    # FROM metadata m
    # LEFT JOIN camera_installation_details c ON m.cameraid = c.cameraid
    # """
    # df_left = pd.read_sql_query(query_left, conn)
    
    # # Second LEFT JOIN: camera_installation_details LEFT JOIN metadata
    # query_right = """
    # SELECT c.SR, c.AC_Name, c.Part_No_Polling_Station_Name, c.Technician_Contact_Number, c.Technican_Name, c.zone, 
    #        m.id, m.cameraid as cameraid, m.timestamp, m.imageurl, m.personcount
    # FROM camera_installation_details c
    # LEFT JOIN metadata m ON c.cameraid = m.cameraid
    # WHERE m.cameraid IS NULL  --Only include rows that were not in the first result set
    # """
    # df_right = pd.read_sql_query(query_right, conn)
    
    # # Concatenate the results to simulate a FULL OUTER JOIN
    # df_full_join = pd.concat([df_left, df_right], ignore_index=True)
    
    # conn.close()
    # return df_full_join
    """Fetch combined data mimicking a FULL OUTER JOIN using LEFT and RIGHT JOINs."""
    conn = sqlite3.connect('metadata.db')
    query = """
    SELECT c.SR, c.AC_Name, c.Part_No_Polling_Station_Name, c.Technician_Contact_Number, c.Technican_Name, c.zone,
           m.id, m.cameraid as metadata_cameraid, m.timestamp, m.imageurl, m.personcount
    FROM camera_installation_details c
    LEFT JOIN metadata m ON c.cameraid = m.cameraid
    UNION ALL
    SELECT c.SR, c.AC_Name, c.Part_No_Polling_Station_Name, c.Technician_Contact_Number, c.Technican_Name, c.zone,
           m.id, m.cameraid as metadata_cameraid, m.timestamp, m.imageurl, m.personcount
    FROM metadata m
    LEFT JOIN camera_installation_details c ON m.cameraid = c.cameraid
    WHERE c.cameraid IS NULL
    """
    try:
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        logging.error("Failed to fetch or parse data", exc_info=True)
        return pd.DataFrame()
    finally:
        conn.close()

llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
agent = create_pandas_dataframe_agent(llm, pd.DataFrame(), agent_type="openai-tools", verbose=True)

def create_prompt(data_frame, user_query, session_history):
    """
    Create a detailed prompt for the LangChain agent.
    
    :param data_frame: The DataFrame containing the joined data from the database.
    :param user_query: The user's query as received from the HTTP request.
    :param session_history: List of past user queries and agent responses to provide context.
    :return: A string that forms the complete prompt to be fed into the model.
    """
    # Build context from the session history
    historical_context = "\n".join(f"User: {x['user']}\nAgent: {x['agent']}" for x in session_history)

    # Instructions for the agent
    instructions = (
        "You are an intelligent agent assisting a user by answering questions based on the provided data. "
        "The data includes camera installation details and metadata about images captured by these cameras. "
        "Your responses should use this data to provide accurate and helpful information. "
        "If the data for a specific query is not available, advise on potential next steps or suggest alternative queries. "
        "Respond in clear, concise language."
        "You have access to metadata about images and detailed camera installation records. "
        "Respond using camera installation details when the query is about SR, AC Name, Part Number, "
        "Technician Contact Number, Technician Name, camera ID, or zone. "
        "If the data for a specific query is not available in the metadata but available in camera installation details, "
        "use the latter to provide an accurate and helpful response. "
        "For other data types or if a query is ambiguous about the data source, use your best judgment to fetch from either table. "
        "You are an intelligent assistant. Use the conversation history and the available data "
        "to provide relevant and accurate information. Consider previous queries and responses "
        "to understand the context of the current question. If data is missing, suggest alternative queries or solutions."
    )

    # Formulate the final prompt
    prompt = f"{historical_context}\n\n{instructions}\n\nUser Query: {user_query}\nResponse:"
    return prompt

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    user_input = data.get('query')
    
    # Load fresh data from the database for each query
    df_used = fetch_full_outer_join_data()
    
    
    
    # Fetch the ongoing conversation from the session, if any
    conversation = session.get('conversation', [])
    context = [item['user'] + '\n' + item['agent'] for item in conversation]  # Gather past dialogues as context
  
    # Create an agent using the selected DataFrame
    agent = create_pandas_dataframe_agent(llm, df_used, agent_type="openai-tools", verbose=True, return_intermediate_steps=True, number_of_head_rows=df_used.shape[0])
    prompt = create_prompt(df_used, user_input, conversation)
    response = agent.invoke({"input": prompt})['output']
    # Invoke the agent to get a response
    # response = agent.invoke({"input": user_input, "context": context})['output']
    print(response)  # Debug: Print the response to the console.
    
    # Append the new interaction to the conversation history
    conversation.append({"user": user_input, "agent": response})
    session['conversation'] = conversation
    
    return jsonify({"user": user_input, "agent": response})


@app.route('/export', methods=['GET'])
def export_conversation():
    conversation = session.get('conversation', [])
    if not conversation:
        return "No conversation to export", 400

    # Return the conversation as a JSON response
    return jsonify(conversation)

if __name__ == '__main__':
    app.run(debug=True, port=7000, host='0.0.0.0')
