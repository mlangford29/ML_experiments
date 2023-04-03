import xgboost as xgb
import numpy as np
import pandas as pd
import featuretools as ft
import mlens
import dask_ml
from tpot import TPOTClassifier

# load the credit card fraud dataset
df = pd.read_csv("creditcard.csv")

# drop the 'Time' column as it is not useful for our analysis
df = df.drop(['Time'], axis=1)

# rename the 'Class' column to 'Fraud'
df = df.rename(columns={'Class': 'Fraud'})

# check for any missing values and drop rows with missing values
if df.isnull().sum().sum() > 0:
    df = df.dropna()

# drop any duplicate rows
if df.duplicated().sum() > 0:
    df = df.drop_duplicates()

# create an entity set and add the transactions entity
es = ft.EntitySet(id='transactions')
es.entity_from_dataframe(entity_id='transactions',
                         dataframe=transactions,
                         make_index=True,
                         index='transaction_id')

# define the primitive transformations to use for deep feature synthesis
agg_primitives = ['sum', 'max', 'min', 'mean', 'count', 'std']
trans_primitives = ['month', 'hour', 'day', 'weekend', 'cum_sum']

# run deep feature synthesis with the defined primitives
features, feature_names = ft.dfs(entityset=es,
                                 target_entity='transactions',
                                 agg_primitives=agg_primitives,
                                 trans_primitives=trans_primitives,
                                 max_depth=3,
                                 n_jobs=-1,
                                 verbose=1)

# Separate features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Set up XGBoost parameters
params = {
    "objective": "binary:logistic",
    "booster": "gbtree",
    "eval_metric": "auc",
    "eta": 0.1,
    "max_depth": 5,
    "subsample": 0.5,
    "colsample_bytree": 0.5,
    "silent": 1,
    "seed": 0
}

# Set up number of runs and create empty feature importances array
n_runs = 10
feat_imp = np.zeros(X.shape[1])

# Run feature importance extraction loop
for i in range(n_runs):
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    
    # Train XGBoost model
    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(params, dtrain, num_boost_round=100)
    
    # Get feature importances
    importance = model.get_score(importance_type='gain')
    
    # Add to running total
    for k in importance.keys():
        feat_imp[int(k[1:])] += importance[k]

# Average feature importances
feat_imp /= n_runs

# grab the top 10 features
importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
top_features = [importance[0] for importance in importances][:10]

# Import necessary libraries
import numpy as np
import pandas as pd
from keras.layers import Input, Dense, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load preprocessed credit card fraud dataset
df = pd.read_csv('credit_card_fraud.csv')

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(['Class'], axis=1), df['Class'], test_size=0.2, random_state=42)

# Get top 10 features using XGBoost feature importance
top_10_features = ['V14', 'V4', 'V10', 'V12', 'V11', 'V17', 'V16', 'V3', 'V7', 'V9']

# Select only top 10 features from dataset
X_train = X_train[top_10_features]

# Define generator neural network
def generator():
    noise_shape = (10,)
    input_layer = Input(shape=noise_shape)
    
    # Hidden layers
    x = Dense(256)(input_layer)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    # Output layer
    output_layer = Dense(len(top_10_features), activation='tanh')(x)
    
    model = Model(inputs=[input_layer], outputs=[output_layer])
    
    return model

# Define discriminator neural network
def discriminator():
    input_layer = Input(shape=(len(top_10_features),))
    
    # Hidden layers
    x = Dense(1024)(input_layer)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    # Output layer
    output_layer = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=[input_layer], outputs=[output_layer])
    
    return model

# Define generator and discriminator models
generator_model = generator()
discriminator_model = discriminator()

# Define combined model to train generator and discriminator simultaneously
def combined(generator, discriminator):
    discriminator.trainable = False
    
    input_layer = Input(shape=(10,))
    
    # Generate synthetic data using generator
    gen_output = generator(input_layer)
    
    # Determine whether synthetic data is real or fake using discriminator
    dis_output = discriminator(gen_output)
    
    model = Model(inputs=[input_layer], outputs=[dis_output])
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    
    return model

# Define number of epochs and batch size
epochs = 3000
batch_size = 128

# Generate synthetic positive cases using generative adversarial network
for i in range(10):
    # Define empty array to store synthetic data
    synthetic_data = np.array([]).reshape(0, len(top_10_features))
    
    # Train combined model
    gan_model = combined(generator_model, discriminator_model)
    for epoch in range(epochs):
        # Generate random noise
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        # Generate synthetic data
        synthetic = generator_model.predict(noise)
        # Combine synthetic and real data
        X = np.concatenate((X_train[top_10_features], synthetic))
        y = np.concatenate((y_train, np.ones((batch_size, 1))))
        # Train discriminator model on combined data
        discriminator_model.trainable = True
        d_loss = discriminator_model.train_on_batch(X, y)
        # Train generator via combined model
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        y = np.zeros((batch_size, 1))
        discriminator_model.trainable = False
        g_loss = gan_model.train_on_batch(noise, y)
        
    # Generate synthetic data using trained generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    synthetic = generator_model.predict(noise)
    # Append synthetic data to synthetic_data array
    synthetic_data = np.vstack([synthetic_data, synthetic])

# Combine original data with synthetic data
combined_data = np.concatenate((X_train, synthetic_data), axis=0)
combined_labels = np.concatenate((y_train, np.ones(len(synthetic_data))))

# Split data into training and validation sets
X_train_combined, X_val_combined, y_train_combined, y_val_combined = train_test_split(combined_data, combined_labels, test_size=0.2, random_state=42)

# Fit linear model
linear_model = LinearRegression()
linear_model.fit(X_train_combined, y_train_combined)

# Evaluate linear model on validation set
linear_model_score = linear_model.score(X_val_combined, y_val_combined)

from scipy.stats import randint, uniform

tpot_config = {
    'sklearn.ensemble.StackingClassifier': {
        'meta_learner': [
            mlens.ensemble.SuperLearner(scorer=metrics.f1_score, 
                                        random_state=42, 
                                        folds=5, 
                                        shuffle=True, 
                                        verbose=1),
            XGBClassifier(),
            LogisticRegression()
        ],
        'n_jobs': [-1],
        'passthrough': [True],
        'stack_method': ['auto'],
        'verbose': [0],
        'use_features_in_secondary': [True],
        'average_probas': [False],
        'shuffle': [True],
        'classifiers': [
            {'n_neighbors': randint(1, 50), 'weights': ['uniform', 'distance']},
            {'criterion': ['gini', 'entropy'], 'max_depth': randint(1, 50), 'min_samples_split': randint(2, 20)},
            {'n_estimators': randint(10, 200), 'max_depth': randint(1, 50), 'min_samples_split': randint(2, 20)},
            {'n_estimators': randint(10, 200), 'learning_rate': uniform(0.01, 0.3), 'max_depth': randint(1, 50), 'min_samples_split': randint(2, 20)},
            {'n_estimators': randint(10, 200), 'learning_rate': uniform(0.01, 0.3), 'base_estimator': [DecisionTreeClassifier(max_depth=d) for d in [1, 2, 3, 4, 5]]},
            {'C': uniform(0.1, 10), 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': ['scale', 'auto', uniform(0.01, 0.1)]},
            {'C': uniform(0.1, 10), 'loss': ['hinge', 'squared_hinge']},
            {'hidden_layer_sizes': [(50,), (100,), (50, 50,), (100, 100,)], 'activation': ['relu', 'tanh'], 'alpha': uniform(0.0001, 0.1)},
            {'n_estimators': randint(10, 200), 'max_depth': randint(1, 50), 'min_samples_split': randint(2, 20)},
            {'alpha': uniform(0.1, 10), 'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']},
            {'n_estimators': randint(10, 200), 'max_samples': uniform(0.1, 1.0), 'max_features': uniform(0.1, 1.0)},
            {'n_estimators': randint(10, 200), 'max_depth': randint(1, 50), 'learning_rate': uniform(0.01, 0.3), 'num_leaves': randint(2, 100)},
            {'depth': randint(1, 8), 'learning_rate': uniform(0.01, 0.3), 'iterations': randint(10, 200), 'l2_leaf_reg': uniform(0.1, 10)}
        ],
        'meta_c': [uniform(0.1, 10)],
        'meta_fit_params': [None],
        'n_folds': [5]
    }
}

# Instantiate TPOT classifier with necessary parameters
tpot = TPOTClassifier(
    generations=5, population_size=20, verbosity=2, n_jobs=-1,
    config_dict='topt_config', use_dask=True, 
    ensemble_size=5, ensemble_nbest=5
)

# Perform the evolutionary search with TPOT
tpot.fit(X_train_combined, y_train_combined)
top_model = tpot.fitted_pipeline_
top_model.export('top_model.py')

# Predict probabilities for the linear model
y_pred_linear = linear_model.predict_proba(X_test)[:, 1]

# Predict probabilities for the ensemble model
y_pred_ensemble = top_model.predict_proba(X_test)[:, 1]

# Calculate AUC for the linear model
auc_linear = roc_auc_score(y_test, y_pred_linear)

# Calculate AUC for the ensemble model
auc_ensemble = roc_auc_score(y_test, y_pred_ensemble)

# Print AUC scores for comparison
print(f"Linear Model AUC: {auc_linear}")
print(f"Ensemble Model AUC: {auc_ensemble}")




