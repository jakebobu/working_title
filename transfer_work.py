import pickle
import pandas as pd
import numpy as np
from words_to_vals import NMF_Time, _tokenize
from work_with_counts import Count_Worker
import datetime as dt
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt

def generate_model_data_from_articles ():
    df = pd.read_csv('temp_data1.csv',index_col=0)
    df = df[df['news_source'] == 'NYT']
    testy = NMF_Time()
    testy.generate_topics(df['content'].values, tok=_tokenize, min_df = 0.01, max_features = 10000, n_components=500)
    testy.perform_time_counting_self(df, delta=dt.timedelta(hours=4), threshold=0.05)
    testy.save_model()

def feature_variance_things():
    df = pd.read_csv('temp_data1.csv',index_col=0)
    df = df[df['news_source'] == 'NYT']
    testy = NMF_Time(verbose=False)
    n_feats = np.linspace(10,1000,10).astype(int)
    errors = []
    in_err = []
    out_err = []
    print('Vectorizing')
    testy._fit_vectorizer(df['content'].values, tok=_tokenize, min_df = 0.01, max_features = 10000)
    print('Range of Features:')
    print(n_feats)
    for n in n_feats:
        print('{} features'.format(n))
        testy._fit_factorizer(n_components=n)
        errors.append(testy.nmf.reconstruction_err_)
        in_similiarity = 0
        for i in range(testy.W.shape[1]):
            topic_articles = testy.W[np.argmax(testy.W,axis=1)==i]
            if len(topic_articles) > 1:
                sim = pairwise_distances(topic_articles,metric='cosine')
                ind = np.tril_indices(topic_articles.shape[0],k=-1)
                in_similiarity += np.sum(sim[ind])
        in_err.append(in_similiarity / testy.W.shape[0])
        H = testy.nmf.components_
        across_sim = pairwise_distances(H,metric='cosine')
        ind = np.tril_indices(H.shape[0],k=-1)
        out_err.append(np.mean(across_sim[ind]))
    # plt.close('all')
    # plt.subplot(3,1,1)
    # plt.title('Reconstruction Errors')
    # plt.plot(n_feats,errors)
    # plt.subplot(3,1,2)
    # plt.title('Average In Topic Similarity')
    # plt.plot(n_feats,in_err)
    # plt.subplot(3,1,3)
    # plt.title('Average Across Topic Similarity')
    # plt.plot(n_feats,out_err)
    # plt.xlabel('Number of Features')
    # plt.show()
    df = pd.DataFrame(columns=['n_feats','reconstruction_error','in_similarity','across_similiarity'])
    df['n_feats']= n_feats
    df['reconstruction_error']= errors
    df['in_similarity'] = in_err
    df['across_similiarity'] = out_err
    df.to_csv('nmf_trends.csv')

if __name__ == '__main__':
    errors, in_err, out_err, n_feats = feature_variance_things()
    print(errors)

    # testy = generate_model_data_from_articles()
    # obj = pull_up_data()
