from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def preprocessing_function(dataset_name, X, y, label_name, batch_size=512, categorical_cols=None, numerical_cols=None):
    
    scaler = StandardScaler()
    # Join X and y into a single dataframe
    df = pd.DataFrame(X)
    df[label_name] = y
    
    # Check if the dataset contains missing values or NaNs
    if df.isnull().values.any(): # Treat missing values
        # Since some of the datasets contains ? values, we replace them with nan and we drop them
        df = df.replace('?', np.nan)
        df.isnull().sum()
        df[pd.isnull(df).any(axis=1)]
        df.dropna(inplace=True)


    if dataset_name=='adult':
        df.drop('education-num', axis=1, inplace=True)

        # Check unique values in label columns
        df[label_name].nunique()
        # Convert <=50K. to <=50K and >50K. to >50K
        df['income'] = df['income'].apply(lambda x: x.replace('.', ''))

        # Encode the label column
        df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

    elif dataset_name=='diabetes':
        df.drop('Education', axis=1, inplace=True)
        df.drop('Income', axis=1, inplace=True)

    elif dataset_name=='breast_cancer':
        # Encode the label column
        df['Class'] = df['Class'].map({'B': 0, 'M': 1})
    elif dataset_name=='shuttle':
        # For 'Class' column each value has to be value-1
        df['Class'] = df['Class'].apply(lambda x: x-1)
    elif dataset_name=='dry_bean':
        df['Class'] = df['Class'].map({'SEKER': 0, 'BARBUNYA': 1, 'BOMBAY': 2, 'CALI': 3, 'DERMASON':4, 'HOROZ': 5, 'SIRA': 6})
    elif dataset_name=='gamma':
        df['Class'] = df['Class'].map({'g': 0, 'h': 1})

    if categorical_cols is not None:
        # One-hot encoding
        df = pd.get_dummies(df, columns=categorical_cols, dtype=int)

    if numerical_cols is not None:
        # Normalize the numerical columns
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    X = df.drop(label_name, axis=1)
    y = df[label_name]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)


    # Convert the numpy arrays to torch tensors
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    # Dataloader for training and testing 
    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    test_data = torch.utils.data.TensorDataset(X_test, y_test)

    # DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

    shape_dataset = X_train.shape[1]

    return  X_train, X_test, y_train, y_test, train_loader, test_loader, shape_dataset






def prepare_dataset(name):
    if name=='adult':
        # Read the Adult dataset from ucimlrepo
        adult = fetch_ucirepo('adult')
        X = adult.data.features
        y = adult.data.targets
        categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
        numerical_cols = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
        X_train, X_test, y_train, y_test, train_loader, test_loader, shape_dataset = preprocessing_function('adult', X, y, 'income', categorical_cols=categorical_columns, numerical_cols=numerical_cols)
    elif name=='diabetes':
        # Read the Adult dataset from ucimlrepo
        diabetes = fetch_ucirepo(id=891) 
        X = diabetes.data.features
        y = diabetes.data.targets
        numerical_cols = ['BMI']
        X_train, X_test, y_train, y_test, train_loader, test_loader, shape_dataset= preprocessing_function('diabetes', X, y, 'Diabetes_binary', numerical_cols=numerical_cols)
    elif name=='shuttle':
        # fetch dataset 
        statlog_shuttle = fetch_ucirepo(id=148) 
        
        # data (as pandas dataframes) 
        X = statlog_shuttle.data.features 
        y = statlog_shuttle.data.targets 
        numerical_cols = X.columns
        X_train, X_test, y_train, y_test, train_loader, test_loader, shape_dataset = preprocessing_function('shuttle', X, y, 'Class', numerical_cols=numerical_cols)
    elif name=='poker':
        # fetch dataset 
        poker_hand = fetch_ucirepo(id=158) 
        
        # data (as pandas dataframes) 
        X = poker_hand.data.features 
        y = poker_hand.data.targets
        numerical_cols = X.columns
        X_train, X_test, y_train, y_test, train_loader, test_loader, shape_dataset = preprocessing_function('poker', X, y, 'Class', numerical_cols=numerical_cols)
    elif name=='spam':
        # fetch dataset 
        spambase = fetch_ucirepo(id=94) 
        
        # data (as pandas dataframes) 
        X = spambase.data.features 
        y = spambase.data.targets 
        numerical_cols = X.columns
        X_train, X_test, y_train, y_test, train_loader, test_loader, shape_dataset = preprocessing_function('spam', X, y, 'Class', numerical_cols=numerical_cols)
    elif name=='musk':
        musk_version_2 = fetch_ucirepo(id=75) 
        # data (as pandas dataframes) 
        X = musk_version_2.data.features 
        y = musk_version_2.data.targets 
        numerical_cols = X.columns
        X_train, X_test, y_train, y_test, train_loader, test_loader, shape_dataset = preprocessing_function('musk', X, y, 'class', numerical_cols=numerical_cols)
    elif name=='dry_bean':
        dry_bean = fetch_ucirepo(id=602) 
        # data (as pandas dataframes) 
        X = dry_bean.data.features 
        y = dry_bean.data.targets
        numerical_cols = X.columns
        X_train, X_test, y_train, y_test, train_loader, test_loader, shape_dataset = preprocessing_function('dry_bean', X, y, 'Class', numerical_cols=numerical_cols)
    elif name=='gamma':
        # fetch dataset 
        magic_gamma_telescope = fetch_ucirepo(id=159) 
        
        # data (as pandas dataframes) 
        X = magic_gamma_telescope.data.features 
        y = magic_gamma_telescope.data.targets 
        numerical_cols = X.columns
        X_train, X_test, y_train, y_test, train_loader, test_loader, shape_dataset = preprocessing_function('gamma', X, y, 'Class', numerical_cols=numerical_cols)
    elif name=='breast_cancer':
        # fetch dataset 
        breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
        
        # data (as pandas dataframes) 
        X = breast_cancer_wisconsin_diagnostic.data.features 
        y = breast_cancer_wisconsin_diagnostic.data.targets
        numerical_cols = X.columns
        X_train, X_test, y_train, y_test, train_loader, test_loader, shape_dataset = preprocessing_function('breast_cancer', X, y, 'Class', numerical_cols=numerical_cols)
    
    return train_loader, test_loader, shape_dataset
