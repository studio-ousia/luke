import pkg_resources
import re


class NLTKSentenceTokenizer(object):
    def __init__(self):
        import nltk
        punkt_param = nltk.tokenize.punkt.PunktParameters()
        punkt_param.abbrev_types = {'dr', 'vs', 'mr', 'bros', 'mrs', 'prof', 'jr', 'inc', 'i.e', 'e.g', 'et al'}
        self._sentence_tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer(punkt_param)

    def __reduce__(self):
        return self.__class__, tuple()

    def span_tokenize(self, text):
        return list(self._sentence_tokenizer.span_tokenize(text))


class JapaneseSentenceTokenizer(object):

    def __init__(self):
        self.pattern = re.compile(r".*?[。|？]")

    def __reduce__(self):
        return self.__class__, tuple()

    def span_tokenize(self, text):
        return [match.span() for match in re.finditer(self.pattern, text)]


class OpenNLPSentenceTokenizer(object):
    _java_initialized = False

    def __init__(self):
        self._initialized = False

    def __reduce__(self):
        return self.__class__, tuple()

    def initialize(self):
        # we need to delay the initialization of Java in order for this class to
        # properly work with multiprocessing
        if not OpenNLPSentenceTokenizer._java_initialized:
            import jnius_config
            jnius_config.add_options('-Xrs')
            jnius_config.set_classpath(pkg_resources.resource_filename(__name__, '/resources/opennlp-tools-1.5.3.jar'))
            OpenNLPSentenceTokenizer._java_initialized = True

        from jnius import autoclass

        File = autoclass('java.io.File')
        SentenceModel = autoclass('opennlp.tools.sentdetect.SentenceModel')
        SentenceDetectorME = autoclass('opennlp.tools.sentdetect.SentenceDetectorME')

        sentence_model_file = pkg_resources.resource_filename(__name__, 'resources/en-sent.bin')
        sentence_model = SentenceModel(File(sentence_model_file))
        self._tokenizer = SentenceDetectorME(sentence_model)

        self._initialized = True

    def span_tokenize(self, text):
        if not self._initialized:
            self.initialize()

        return [(span.getStart(), span.getEnd()) for span in self._tokenizer.sentPosDetect(text)]
