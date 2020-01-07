import pandas as pd
import time
from sklearn.model_selection import train_test_split
import numpy as np
#import ray
#import psutil
from evaluation_script import read_file, recall_at_k, precision_at_k
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from multiprocessing import Pool
from functools import partial
import math
from operator import itemgetter



def timeit(method):
    """
    Standard Python decorator that measures the execution time of a method;
    """
    def timed(*args, **kw):
        start = time.time()
        result = method(*args, **kw)
        end = time.time()
        
        print(f"{method.__name__}:  {(end - start):.2f} s")
        return result
    return timed

#############################
#############################

@timeit
def load_training_data(data_path: str, row_subset: float=1, train_split: float=0.7, shuffle: bool=False, seed=None):
    '''
    Load the training set and divide it into training and test splits.
    "LinkedID" is the value that we want to predict
    :param data_path: path to the dataset to load;
    :param row_subset: use only the specified fraction of rows in the dataset (value in (0, 1]);
    :param train_split: fraction of rows placed in the training set;
    :param shuffle: if True, shuffle the rows before splitting or subsetting the data;
    '''
    if row_subset <= 0 or row_subset > 1:
        row_subset = 1
    
    data = read_file(training_file, set_record_id_as_index=True)
    if shuffle:
        data = data.sample(frac=1, random_state=seed)
    # Obtain the specified subset of rows;
    data = data.iloc[:int(np.ceil(len(data) * row_subset))]
        
    X = data.drop(columns="linked_id")
    y = data["linked_id"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_split, shuffle=shuffle, random_state =seed)
    
    return X_train, X_test, y_train, y_test

#############################
#############################

@timeit
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform cleaning of the dataset, e.g. lowercase strings, fill missing values, etc...
    """
    cleaned_data = data.copy()
    for c in ["name", "type", "address", "phone", "email", "modification"]:
        cleaned_data[c] = cleaned_data[c].str.lower()
    
    return cleaned_data

#############################
#############################
    
@timeit
def index_data(data: list) -> list:
    """
    Manipulate the data to create indices that speed up the computation. 
    
    The dataframes to be indexed are passed with a list of pd.DataFrames.
    Be careful not to leak information from the train dataset to the test dataset!
    
    In this case, replace all strings with unique integer values,
    so that comparing rows for equality is much faster;
    """
    
    # Obtain the original ending of each dataframe.
    # Prepend a 0 to indicate the start of the first dataframe;
    lengths = [0] + [len(x) for x in data]
    lengths = np.cumsum(lengths)
    # Join together all the input data;
    concatenated_data = pd.concat(data)
        
    for c in ["email_domain","type", "address_country","phone_prefix","email_suffix","modification"]:
        concatenated_data[c] = pd.factorize(concatenated_data[c])[0]
    
    # Split the input data;
    indexed_data = []
    for i in range(len(lengths) - 1):
        indexed_data += [concatenated_data.iloc[lengths[i]:lengths[i + 1], :]]
    
    return indexed_data

#############################
#############################


    
def predict_record_fast(train: tuple, record: np.array, K: int=10) -> list:
    """
    Given a record and a training set, find the records in the training set that best match
    the input record. Predictions can be done using very simple unsupervised algorithms,
    as done in this example, or computed with some fancy ML model;
    """
    # In this simple model, look for the number of identical columns for each training record.
    # Skip the last column (type of modification);
    X_train,y_train=train[0],train[1]
    #identical_values_per_row = pd.Series(np.array(record[3:] == X_train[:,3:]).sum(axis=1))

    # Fuzzy name match
    dataframecolumn0 = pd.DataFrame([record[0]])
    dataframecolumn0.columns = ['Match0']
    
    compare0 = pd.DataFrame(X_train[:,0])
    compare0.columns = ['compare0']
    
    dataframecolumn0['Key0'] = 1
    compare0['Key0'] = 1
    combined_dataframe0 = dataframecolumn0.merge(compare0,on="Key0",how="left")
    
    identical_values_per_row = pd.Series(partial_match_vector_name(combined_dataframe0['Match0'],combined_dataframe0['compare0']))

    # Fuzzy address match
    dataframecolumn1 = pd.DataFrame([record[1]])
    dataframecolumn1.columns = ['Match1']
    
    compare1 = pd.DataFrame(X_train[:,1])
    compare1.columns = ['compare1']
    
    dataframecolumn1['Key1'] = 1
    compare1['Key1'] = 1
    combined_dataframe1 = dataframecolumn1.merge(compare1,on="Key1",how="left")
    
    identical_values_per_row += partial_match_vector_address(combined_dataframe1['Match1'],combined_dataframe1['compare1'])

    # Fuzzy email_first_part match
    dataframecolumn2 = pd.DataFrame([record[3]])
    dataframecolumn2.columns = ['Match2']
    
    compare2 = pd.DataFrame(X_train[:,3])
    compare2.columns = ['compare2']
    
    dataframecolumn2['Key2'] = 1
    compare2['Key2'] = 1
    combined_dataframe2 = dataframecolumn2.merge(compare2,on="Key2",how="left")
    
    identical_values_per_row += partial_match_vector_email_first(combined_dataframe2['Match2'],combined_dataframe2['compare2'])

    #Fuzzy email_domain match
    dataframecolumn3 = pd.DataFrame([record[2]])
    dataframecolumn3.columns = ['Match3']
    
    compare3 = pd.DataFrame(X_train[:,2])
    compare3.columns = ['compare3']
    
    dataframecolumn3['Key3'] = 1
    compare3['Key3'] = 1
    combined_dataframe3 = dataframecolumn3.merge(compare3,on="Key3",how="left")
    
    identical_values_per_row += partial_match_vector_number_no_prefix(combined_dataframe3['Match3'],combined_dataframe3['compare3'])
    # Obtain the K rows with the most matches;
    best_matches = identical_values_per_row.sort_values(ascending=False)[:K]    
    # Retrieve the original record IDs from the training set;
    #print(best_matches)
    return list(zip(list(y_train.loc[best_matches.index.values]) ,list(best_matches)))

@timeit
def predict_er_fast(X_train, y_train, name,address,phone,mail,soglia, K=10) -> dict:
#def predict_er_fast(train, X_test, K=10) -> dict:
    """
    Given a training dataset and a test dataset, obtain the top-K predictions 
    for each record in the test dataset;
    """

    # Store for each record the list of predictions;
    predictions = {}
    start = time.time()
    
    # Extract the matrix that represent the data.
    # Skip the last column;
    #X_train_matrix = X_train.values[:, :-1]
    
    # Also reindex the y column, as we lose the original index when doing a comparison using matrices;
    y_train_vector = y_train.reset_index(drop=True)
    #X_train=X_train.reset_index(drop=True)
    n_threads=18
    n_reduced=math.ceil(len(X_train.name)/float(n_threads))
    chunks=[X_train.iloc[i*n_reduced:(i+1)*n_reduced,1:].values for i in range(n_threads)]
    chunks_y=[y_train_vector.iloc[i*n_reduced:(i+1)*n_reduced].reset_index(drop=True) for i in range(n_threads)]
    print(chunks[0])
    # Compute a prediction for each record;
    i=0
    for i in range(1): #,X_test['email_domain'],X_test['type'],X_test['address_country'],X_test['phone_prefix'],X_test['email_suffix']):
        # Extract values from the current row;
        row_i=(name,address,phone,mail)
        row_i_vector = np.array(row_i)
        print(row_i_vector)
        # Find the best matching record for the current record;
        pool=Pool(n_threads)
        input_data = tuple(zip(chunks,chunks_y))
        appoggio =[x[:] for x in pool.map(partial(predict_record_fast, record=row_i_vector,K=10),input_data)]
        appoggio_2=[item for sublist  in appoggio for item in sublist]
        #print(appoggio_2)
        appoggio_3=sorted(appoggio_2,key=lambda x : float(x[1]),reverse=True)
        predictions[0]=[i for i in appoggio_3 if i[1]>=soglia][:K]
    return predictions


@timeit
def prediction_dict_to_df(predictions: dict) -> pd.DataFrame:
    # Turn the prediction dict into a series of tuples;
    results = []
    for query_id, pred_list in predictions.items():
        for p in pred_list:
            results += [[query_id, p]]
    return pd.DataFrame(results, columns=["queried_record_id", "predicted_record_id"])

@timeit
def prediction_dict_to_kaggle_df(predictions: dict) -> pd.DataFrame:
    # Turn the prediction dict into a series of tuples;
    results = []
    for query_id, pred_list in predictions.items():
        results += [[query_id, " ".join(pred_list)]]
    return pd.DataFrame(results, columns=["queried_record_id", "predicted_record_id"])

@timeit
def kaggle_sol_to_df(kaggle_df: pd.DataFrame) -> pd.DataFrame:
    kaggle_df_indexed = kaggle_df.set_index("queried_record_id")
    results = []
    for query_id, pred_list in kaggle_df_indexed.iterrows():
        results += [[query_id, pred] for pred in pred_list["predicted_record_id"].split(" ")]
    return pd.DataFrame(results, columns=["queried_record_id", "predicted_record_id"])

#############################
#############################
#Load data set


training_filez = "../panama-papers-polimi/data/entity-resolution_advanced-topics-training_data.csv"
train = read_file(training_filez, set_record_id_as_index=False)
#X_train_parra = train.drop(columns="linked_id")
y_train = train["record_id"]
#X_train, X_test, y_train, y_test = load_training_data(training_file, shuffle=True, row_subset=0.01, seed=42)

trainn=pd.read_csv('../panama-papers-polimi/data/panama_train_expanded_2.csv')
#trainn = read_file(training_file, set_record_id_as_index=True)

testt=pd.read_csv('../panama-papers-polimi/data/panama_train_expanded_2.csv')
#testt = read_file(testing_file, set_record_id_as_index=True)

X_train=trainn.copy()
X_test=testt.copy()

X_train=X_train[["record_id","name","address","email_first_part","number_no_prefix"]] #"email_domain","type", "address_country","phone_prefix","email_suffix","modification"]]
X_test=X_test[["record_id","name","address","email_first_part","number_no_prefix"]] #"email_domain","type" ,"address_country","phone_prefix","email_suffix","modification"]]

#X_train,X_test=index_data([X_train,X_test])
#X_train=X_train[["name","address","email_first_part","number_no_prefix","email_domain","type", "address_country","phone_prefix","email_suffix","modification"]]
#X_test=X_test[["record_id","name","address","email_first_part","number_no_prefix","email_domain","type" ,"address_country","phone_prefix","email_suffix","modification"]]
X_train.number_no_prefix=X_train.number_no_prefix.apply(str)
X_test.number_no_prefix=X_test.number_no_prefix.apply(str)

X_train=X_train.fillna('-1')
X_test=X_test.fillna('-1')

X_train.address[X_train.address.str.len()<=1]='-1'
X_train.email_first_part[X_train.email_first_part.str.len()<=1]='-1'
X_train.number_no_prefix[X_train.number_no_prefix.str.len()<=1]='-1'

#test.name[test.name.str.len()<=1]='-1'
X_test.address[X_test.address.str.len()<=3]='-1'
X_test.email_first_part=X_test.email_first_part.apply(lambda x : '-1' if len(x)<=2 else x)

def partial_match_name(x,y):
    return(fuzz.ratio(x,y)*0.092)
partial_match_vector_name = np.vectorize(partial_match_name)

def partial_match_address(x,y):
    return(fuzz.ratio(x,y)*0.002)
partial_match_vector_address = np.vectorize(partial_match_address)

def partial_match_email_first(x,y):
    return(fuzz.ratio(x,y)*0.002)
partial_match_vector_email_first = np.vectorize(partial_match_email_first)

def partial_match_number_no_prefix(x,y):
    return(fuzz.ratio(x,y)*0.004)
partial_match_vector_number_no_prefix = np.vectorize(partial_match_number_no_prefix)

print('Welcome,please type the personal information of the person you need to check')

name = input('NAME :')
address = input('ADDRESS :')
phone= input('PHONE :')
mail=input('MAIL :')

soglia = input('Please insert a threshold value between 0 and 10 :')
K=input('Lastly, please insert the number of matching records you would like to retrieve :')
predictions = predict_er_fast(X_train, y_train,name,address,phone,mail,float(soglia),int(K))
print('This is the list of the best matching candidates and their similarity scores you asked for : ')
print(predictions[0])
array=[]
for j in predictions[0]:
    if j[0]!='':
    	array.append(j[0])

print('And now you can check the personal information of the best matches : ')
#print(X_train)
X_train.set_index('record_id',inplace=True)
print(X_train.loc[array,'name'])



'''
#soglia='8.0'
pred_df = prediction_dict_to_df(predictions)

csv_name = 'results_con_soglia_a_'+ str(soglia)
csv_name += datetime.now().strftime('%b%d_%H-%M-%S')+'.csv'
#pred_df.to_csv("../panama-oracle-f-2/"+csv_name, index=False)

pred_df_kaggle = prediction_dict_to_kaggle_df(predictions)

pred_df_kaggle.to_csv("../panama-papers-polimi/"+csv_name, index=False)
'''
