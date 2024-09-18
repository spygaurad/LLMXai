import pandas as pd
import csv
import json

def model_question_format():
    df = pd.read_csv('model_questions.csv')
    df.columns = ['model_name', 'layer', 'Question']

    df['conversation_type'] = 'independent'
    df['question_type'] = 'domain'
    print(df.head())
    return df

def independent_q_format():
    df = pd.read_csv('independent_clean.csv')
    df.columns = ['Question']

    df['model_name'] = 'empty'
    df['layer'] = 'empty'
    df['conversation_type'] = 'independent'
    df['question_type'] = 'general'
    print(df.head())
    return df

def continuation_q_format():
    df = pd.read_csv('continuation_clean.csv')
    df.columns = ['Question']

    df['model_name'] = 'empty'
    df['layer'] = 'empty'
    df['conversation_type'] = 'continuation'
    df['question_type'] = 'general'
    print(df.head())
    return df

df1 = model_question_format()
df2 = independent_q_format()
df3 = continuation_q_format()

merged_df = pd.concat([df1, df2, df3], ignore_index=True)

merged_df['Answer'] = merged_df.apply(lambda row: json.dumps({
    'model_name': row['model_name'],
    'layer': row['layer'],
    'conversation_type': row['conversation_type'],
    'question_type': row['question_type']
}), axis=1)

merged_df.to_csv('Merged_data.csv', index=False)