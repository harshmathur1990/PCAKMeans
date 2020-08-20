for index, i in enumerate(not_selected_frames):
    print ('{} - {}\n'.format(index, i))
    seldata = np.transpose(data[i, 0], axes=(1, 2, 0))
    seldata = seldata.reshape(1236 * 1848, 30)
    avg_4000 = np.mean(seldata[:, -1])
    seldata = seldata / avg_4000

    full_mean = np.mean(seldata, axis=0)
    full_mean = full_mean[np.newaxis, :]

    full_std = np.std(seldata, axis=0)
    full_std = full_std[np.newaxis, :]

    meanr = np.repeat(full_mean, axis=0, repeats=seldata.shape[0])
    stdr = np.repeat(full_std, axis=0, repeats=seldata.shape[0])

    zscore = np.divide(np.subtract(seldata, meanr), stdr)

    zscore_rest[index * 1236 * 1848: (index + 1) * 1236 * 1848, :] = zscore


for index, i in enumerate(not_selected_frames):
    model = KMeans(n_clusters=45)
    model.labels_ = out_45['labels_'][()]
    model.cluster_centers_ = out_45['cluster_centers_'][()]
    model.inertia_ = out_45['inertia_'][()]
    model.n_iter_ = out_45['n_iter_'][()]

    labels = model.predict(zscore_rest[index * 1236 * 1848: (index + 1) * 1236 * 1848, :])

    labels = labels.reshape(1236, 1848)

    label_memmap[index] = labels