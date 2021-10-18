import numpy
from Common import tokenize, tf_idf


def run(docs):
    new_docs = tokenize(docs)
    vocabulary, idf, train_matrix = tf_idf(new_docs)
    centroids = [numpy.random.random(len(vocabulary)) for _ in range(4)]
    clusters = [[] for _ in range(4)]
    train_matrix = numpy.asarray(train_matrix)
    temp = [row @ row for row in train_matrix]
    x2 = numpy.asarray([[temp[i] for _ in range(4)] for i in range(len(docs))])
    for i in range(100):
        for j in range(4):
            clusters[j].clear()
        xy = train_matrix @ numpy.transpose(centroids)
        temp = [row @ row for row in numpy.asarray(centroids)]
        y2 = numpy.asarray([temp for _ in range(len(docs))])
        distance = list(x2 - 2 * xy + y2)
        # for j in range(len(new_docs)):
        #     clusters[list(distance[j]).index(min(list(distance[j])))].append(j)

        for j in range(4):
            sum_of_vectors = [0 for _ in range(len(vocabulary))]
            sum_of_vectors = [0 for _ in range(len(vocabulary))]

            for k in clusters[j]:
                sum_of_vectors = numpy.add(sum_of_vectors, train_matrix[k])
            centroids[j] = numpy.asarray(sum_of_vectors / len(clusters[j]) if len(clusters[j]) > 0 else sum_of_vectors)
    print('Done')
