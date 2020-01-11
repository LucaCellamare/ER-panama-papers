import pandas as pd
import time
from sklearn.model_selection import train_test_split
import numpy as np
import ray
import psutil
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from multiprocessing import Pool
from functools import partial
import math
from operator import itemgetter

#fuzzy string matching on name,address,telephone,mail
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

def read_file(path: str, set_record_id_as_index: bool=False) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, escapechar="\\", index_col="record_id" if set_record_id_as_index else None)
#############################
#############################

def recall_at_k(resultTable : pd.DataFrame, trainingData: pd.DataFrame, testingData: pd.DataFrame) -> dict:
    """
    Given a list of K predictions for each query, first retrieve the correct ID from the test data,
    then look in the training data the percentage of records that have been successfully identified.
    
    For example, given query "1234-M", first retrieve the correct ID "1234" from the test data,
    then obtain from the training data all records that refer to "1234", 
    and finally look how many of them we have found;
    """
    
    # Obtain all the predictions for each record in the test set;
    perQueryRecords = resultTable.groupby("queried_record_id")
    
    # Group training records by their LinkedID truth value;
    groupedTrainingRecords = trainingData.groupby("linked_id")

    totalRecall = 0.0

    allRecords = dict()
    
    start = time.time()
    for i, (queriedRecordID, group) in enumerate(perQueryRecords):
        #if i % 1000 == 0 and i > 0:
            #print(f"processed {i}/{len(perQueryRecords)} records, {100 * i / len(perQueryRecords):.2f}%")
            #print(f"\tcurrent recall: {(totalRecall / i):.2f}")
            #print(f"\ttime elapsed: {(time.time() - start):.2f} s")
        
        try:
            queriedLinkedID = testingData.loc[queriedRecordID, "linked_id"]
        except IndexError:
            raise IndexError("ID {queriedRecordID} not found in testing data!")
        
        try:
            allRelevantRecords = set(groupedTrainingRecords.get_group(queriedLinkedID).index.values)
        except KeyError:
            allRelevantRecords = set()
        setPredictedRecords = set(group["predicted_record_id"])
        selectedRelevantRecords = setPredictedRecords.intersection(allRelevantRecords)
        recall = 1
        if (len(allRelevantRecords) > 0):
            recall = len(selectedRelevantRecords) / len(allRelevantRecords)

        totalRecall += recall
        allRecords[queriedRecordID] = [queriedRecordID, recall, len(selectedRelevantRecords), len(allRelevantRecords)]
    
    # Store the results in a summary table;
    result_table =  pd.DataFrame.from_dict(
                        allRecords,
                        orient='index',
                        columns=["QueriedRecordID", "Recall@K", "SelectedRecords", "AllRelevantRecords"]
                    )
    # Compute the filtered recall, which considers only queries with at least one relevant record in the training data;
    queries_with_relevant_records = result_table[result_table["AllRelevantRecords"] > 0]
    filtered_recall = np.mean(queries_with_relevant_records["SelectedRecords"] / queries_with_relevant_records["AllRelevantRecords"])

    return {
            "AverageRecall" : totalRecall / len(perQueryRecords),
            "AverageFilteredRecall": filtered_recall,
            "perQueryResult" : result_table
            }
    
def precision_at_k(resultTable : pd.DataFrame, trainingData: pd.DataFrame, testingData: pd.DataFrame) -> dict:
    """
    Given a list of K predictions for each query, first retrieve the correct ID from the test data,
    then look in the training data the percentage of records that are actually relevant;
    
    For example, given query "1234-M", first retrieve the correct ID "1234" from the test data,
    then obtain from the training data all records that refer to "1234", 
    and finally look how many of the records we have found are actually referring to "1234"
    """
    
    # Obtain all the predictions for each record in the test set;
    perQueryRecords = resultTable.groupby("queried_record_id")
    
    # Group training records by their LinkedID truth value;
    groupedTrainingRecords = trainingData.groupby("linked_id")

    totalPrecision = 0.0
    numberOfPredictionsForRelevantRecords = 0

    allRecords = dict()
    
    start = time.time()
    for i, (queriedRecordID, group) in enumerate(perQueryRecords):
        #if i % 1000 == 0 and i > 0:
            #print(f"processed {i}/{len(perQueryRecords)} records, {100 * i / len(perQueryRecords):.2f}%")
            #print(f"\tcurrent precision: {(totalPrecision / i):.2f}")
            #print(f"\ttime elapsed: {(time.time() - start):.2f} s")
        
        try:
            queriedLinkedID = testingData.loc[queriedRecordID, "linked_id"]
        except IndexError:
            raise IndexError("ID {queriedRecordID} not found in testing data!")
        
        try:
            allRelevantRecords = set(groupedTrainingRecords.get_group(queriedLinkedID).index.values)
        except KeyError:
            allRelevantRecords = set()
        setPredictedRecords = set(group["predicted_record_id"])
        selectedRelevantRecords = setPredictedRecords.intersection(allRelevantRecords)
        precision = 1
        if (len(allRelevantRecords) > 0):
            precision = len(selectedRelevantRecords) / len(setPredictedRecords)
            numberOfPredictionsForRelevantRecords += len(setPredictedRecords)

        totalPrecision += precision
        allRecords[queriedRecordID] = [queriedRecordID, precision, len(selectedRelevantRecords), len(allRelevantRecords)]
    
    # Store the results in a summary table;
    result_table =  pd.DataFrame.from_dict(
                        allRecords,
                        orient='index',
                        columns=["QueriedRecordID", "Precision@K", "SelectedRecords", "AllRelevantRecords"]
                    )
    # Compute the filtered recall, which considers only queries with at least one relevant record in the training data;
    queries_with_relevant_records = result_table[result_table["AllRelevantRecords"] > 0]
    filtered_precision = np.mean(queries_with_relevant_records["SelectedRecords"] / numberOfPredictionsForRelevantRecords)

    return {
            "AveragePrecision" : totalPrecision / len(perQueryRecords),
            "AverageFilteredPrecision": filtered_precision,
            "perQueryResult" : result_table
            }   

#%%

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
        
    #X = data.drop(columns="linked_id")
    y = data[["linked_id"]]
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
def predict_record_fast(train: tuple, record: np.array, K: int=1000) -> list:
    """
    Given a record and a training set, find the records in the training set that best match
    the input record. Predictions can be done using very simple unsupervised algorithms,
    as done in this example, or computed with some fancy ML model;
    """
    # In this simple model, look for the number of identical columns for each training record.
    # Skip the last column (type of modification);
    X_train,y_train=train[0],train[1]
    #identical_values_per_row = pd.Series(np.array(record[3:] == X_train[:,3:]).sum(axis=1))

    # Fuzzy match
    df0 = pd.DataFrame([record[0]])
    df0.columns = ['Match0']
    
    compare0 = pd.DataFrame(X_train[:,0])
    compare0.columns = ['compare0']
    
    df0['Key0'] = 1
    compare0['Key0'] = 1
    combined_dataframe0 = df0.merge(compare0,on="Key0",how="left")
    
    identical_values_per_row = pd.Series(partial_match_vector_name(combined_dataframe0['Match0'],combined_dataframe0['compare0']))
    # Obtain the K rows with the most matches;
    best_matches = identical_values_per_row.sort_values(ascending=False)[:K]    
    # Retrieve the original record IDs from the training set;
    return list(zip(list(y_train.loc[best_matches.index.values]['linked_id'].values) ,list(best_matches)))

@timeit
def predict_er_fast(X_train, y_train, X_test,soglia, K=1000) -> dict:
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
    n_threads=12
    n_reduced=math.ceil(len(X_train.name)/float(n_threads))
    chunks=[X_train.iloc[i*n_reduced:(i+1)*n_reduced,:].values for i in range(n_threads)]
    chunks_y=[y_train_vector.iloc[i*n_reduced:(i+1)*n_reduced].reset_index(drop=True) for i in range(n_threads)]


    # Compute a prediction for each record;
    i=0
    for row_i in zip(X_test['record_id'],X_test['complessivo']):
        row_i_vector = np.array(row_i[1:])
        if i % 1000 == 0 and i > 0:
            print(f"processed {i}/{len(X_test)} records, {100 * i / len(X_test):.2f}%")
            print(f"\ttime elapsed: {(time.time() - start):.2f} s")
        
        # Find the best matching record for the current record;
        pool=Pool(n_threads)
        input_data = tuple(zip(chunks,chunks_y))
        appoggio =[x[:] for x in pool.map(partial(predict_record_fast, record=row_i_vector,K=10),input_data)]
        appoggio_2=[item for sublist  in appoggio for item in sublist]
        appoggio_3=sorted(appoggio_2,key=itemgetter(1),reverse=True)[:10]
        predictions[row_i[0]]=[i[0] for i in appoggio_3 if i[1]>=soglia]
        i+=1
        pool.close()
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


training_filez = "../panama-papers-polimi/data/entity-resolution_advanced-topics-training_data.csv"
train = read_file(training_filez, set_record_id_as_index=True)
        
X_train_parra = train.drop(columns="linked_id")
y_train = train["linked_id"]
trainn=pd.read_csv('../panama-papers-polimi/data/panama_train_expanded_2.csv')
testt=pd.read_csv('../panama-papers-polimi/data/panama_train_expanded_2.csv')

X_train=trainn.copy()
X_test=testt.copy()

X_train=X_train[["name","address","email","phone"]] 
X_test=X_test[["record_id","name","address","email","phone"]] 

X_train.phone=X_train.phone.apply(str)
X_test.phone=X_test.phone.apply(str)


X_train=X_train.fillna('')
X_test=X_test.fillna('') 
X_train.name=X_train.name.apply(lambda x : '' if x=='-1' else x)
X_train.address=X_train.address.apply(lambda x : '' if x=='-1' else x)
X_train.email=X_train.email.apply(lambda x : '' if x=='-1' else x)
X_train.phone=X_train.phone.apply(lambda x : '' if x=='-1' else x)

X_test.name=X_test.name.apply(lambda x : '' if x=='-1' else x)
X_test.address=X_test.address.apply(lambda x : '' if x=='-1' else x)
X_test.email=X_test.email.apply(lambda x : '' if x=='-1' else x)
X_test.phone=X_test.phone.apply(lambda x : '' if x=='-1' else x)

X_train=X_train[['record_id','name','address','email','phone']]
X_train['complessivo']=X_train['name']+' '+X_train['address']+' '+X_train['email']+' '+X_train['phone']
X_train.set_index('record_id',inplace=True)

X_test['complessivo']=X_test['name']+' '+X_test['address']+' '+X_test['email']+' '+X_test['phone']



#Vectorized fuzzy string matching operation in order to calculate all the similarity scores for one test instance in just one step
def partial_match_totale(x,y):
    return(fuzz.partial_token_set_ratio(x,y)*0.1)
partial_match_vector_totale = np.vectorize(partial_match_totale)
soglia='8.0'
predictions = predict_er_fast(X_train, y_train, X_test,int(soglia))

pred_df = prediction_dict_to_df(predictions)
#create kaggle format predictions and calculate metrics

pred_df_kaggle = prediction_dict_to_kaggle_df(predictions)
pred_df_kaggle.to_csv("../panama-papers-polimi/preds/kaggle_fuzzy_multithreaded_partial_token_set_top_10"+'.csv', index=False)
pred_df = prediction_dict_to_df(predictions)

#%% 5. Compute recall@K;
recall_dict = recall_at_k(pred_df, train_m, test_m)
print(recall_dict["AverageRecall"])
print(recall_dict["AverageFilteredRecall"])

#%% 6. Compute MAP@K;
precision_dict = precision_at_k(pred_df, train_m, test_m)
print(precision_dict["AveragePrecision"])
print(precision_dict["AverageFilteredPrecision"])
