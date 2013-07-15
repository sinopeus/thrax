"""
Verbose debug output for the model.
"""

import logging
import stats

import examples

import numpy
import random

def diagnostics(cnt, model):
    logging.info(stats())
    idxs = range(model.parameters.vocab_size)
    random.shuffle(idxs)
    idxs = idxs[:100]

    debug_prehidden_values(cnt, model)
    embeddings_debug(model.parameters.embeddings[idxs], cnt, "rand 100 words")
    embeddings_debug(model.parameters.embeddings[:100], cnt, "top  100 words")
    embeddings_debug(model.parameters.embeddings[model.parameters.vocab_size/2-50:model.parameters.vocab_size/2+50], cnt, "mid  100 words")
    embeddings_debug(model.parameters.embeddings[-100:], cnt, "last 100 words")
    weights_debug(model.parameters.hidden_weights.value, cnt, "hidden weights")
    weights_debug(model.parameters.output_weights.value, cnt, "output weights")
    logging.info(stats())

def visualizedebug(cnt, model, rundir, newkeystr, WORDCNT=500):
    idxs = range(model.parameters.vocab_size)
    random.shuffle(idxs)
    idxs = idxs[:WORDCNT]

    visualize(cnt, model, rundir, idxs, "randomized%s" % newkeystr)
    visualize(cnt, model, rundir, range(WORDCNT), "mostcommon%s" % newkeystr)
    visualize(cnt, model, rundir, range(-1, -WORDCNT, -1), "leastcommon%s" % newkeystr)
    visualize(cnt, model, rundir, range(model.parameters.vocab_size/2-WORDCNT/2,model.parameters.vocab_size/2+WORDCNT/2), "midcommon%s" % newkeystr)

def debug_prehidden_values(cnt, model):
    """
    Give debug output on pre-squash hidden values.
    """
    for (i, ve) in enumerate(examples.get_validation_example()):
        (score, prehidden) = model.verbose_predict(ve)
        abs_prehidden = numpy.abs(prehidden)
        med = numpy.median(abs_prehidden)
        abs_prehidden = abs_prehidden.tolist()
        assert len(abs_prehidden) == 1
        abs_prehidden = abs_prehidden[0]
        abs_prehidden.sort()
        abs_prehidden.reverse()
        logging.info("%s %s %s %s %s" % (cnt, "abs(pre-squash hidden) median =", med, "max =", abs_prehidden[:3]))
        if i+1 >= 3: break

def visualize(cnt, model, rundir, idxs, str):
    """
    Visualize a set of examples using t-SNE.
    """
    from vocabulary import wordmap
    PERPLEXITY=30

    x = model.parameters.embeddings[idxs]
    print(x.shape)
    titles = [wordmap.str(id) for id in idxs]

    import os.path
    filename = os.path.join(rundir, "embeddings-%s-%d.png" % (str, cnt))
    try:
        from textSNE.calc_tsne import tsne
#       from textSNE.tsne import tsne
        out = tsne(x, perplexity=PERPLEXITY)
        from textSNE.render import render
        render([(title, point[0], point[1]) for title, point in zip(titles, out)], filename)
    except IOError:
        logging.info("ERROR visualizing", filename, ". Continuing...")

def embeddings_debug(w, cnt, str):
    """
    Output the l2norm mean and max of the embeddings, including in debug out the str and training cnt
    """
    totalcnt = numpy.sum(numpy.abs(w) >= 0)
    notsmallcnt = numpy.sum(numpy.abs(w) >= 0.1)
    logging.info("%d %s dimensions of %s have absolute value >= 0.1" % (cnt, percent(notsmallcnt, totalcnt), str))
    notsmallcnt = numpy.sum(numpy.abs(w) >= 0.01)
    logging.info("%d %s dimensions of %s have absolute value >= 0.01" % (cnt, percent(notsmallcnt, totalcnt), str))

    l2norm = numpy.sqrt(numpy.square(w).sum(axis=1))
    median = numpy.median(l2norm)
    mean = numpy.mean(l2norm)
    std = numpy.std(l2norm)
#    print("%d l2norm of top 100 words: mean = %f stddev=%f" % (cnt, numpy.mean(l2norm), numpy.std(l2norm),))
    l2norm = l2norm.tolist()
    l2norm.sort()
    l2norm.reverse()
    logging.info("%d l2norm of %s: median = %f mean = %f stddev=%f top3=%s", cnt, str, median, mean, std, repr(l2norm[:3]))
#    print("top 5 = %s" % `l2norm[:5]`)

def weights_debug(w, cnt, str):
    """
    Output the abs median, mean, and max of the weights w, including in debug out the str and training cnt
    """
    w = numpy.abs(w)
    logging.info("%d abs of %s: median=%f mean=%f stddev=%f" % (cnt, str, numpy.median(w), numpy.mean(w), numpy.std(w),))
#    print("%d l2norm of top 100 words: mean = %f stddev=%f" % (cnt, numpy.mean(l2norm), numpy.std(l2norm),))
#    w = w.tolist()
#    w.sort()
#    w.reverse()
#    logging.info("\ttop 5 = %s" % `w[:5]`)
#    print("top 5 = %s" % `l2norm[:5]`)
