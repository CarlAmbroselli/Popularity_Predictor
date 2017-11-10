from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import numpy as np
import re

class Helper():
    # initialize stemmer
    stemmer = SnowballStemmer('german')
    # grab stopword list
    # stop = stopwords.words('german')
    stop = ['a', 'ab', 'aber', 'aber', 'ach', 'acht', 'achte', 'achten', 'achter', 'achtes', 'ag', 'alle', 'allein', 'allem', 'allen', 'aller', 'allerdings', 'alles', 'allgemeinen', 'als', 'als', 'also', 'am', 'an', 'andere', 'anderen', 'andern', 'anders', 'au', 'auch', 'auch', 'auf', 'aus', 'ausser', 'außer', 'ausserdem', 'außerdem', 'b', 'bald', 'bei', 'beide', 'beiden', 'beim', 'beispiel', 'bekannt', 'bereits', 'besonders', 'besser', 'besten', 'bin', 'bis', 'bisher', 'bist', 'c', 'd', 'da', 'dabei', 'dadurch', 'dafür', 'dagegen', 'daher', 'dahin', 'dahinter', 'damals', 'damit', 'danach', 'daneben', 'dank', 'dann', 'daran', 'darauf', 'daraus', 'darf', 'darfst', 'darin', 'darüber', 'darum', 'darunter', 'das', 'das', 'dasein', 'daselbst', 'dass', 'daß', 'dasselbe', 'davon', 'davor', 'dazu', 'dazwischen', 'dein', 'deine', 'deinem', 'deiner', 'dem', 'dementsprechend', 'demgegenüber', 'demgemäss', 'demgemäß', 'demselben', 'demzufolge', 'den', 'denen', 'denn', 'denn', 'denselben', 'der', 'deren', 'derjenige', 'derjenigen', 'dermassen', 'dermaßen', 'derselbe', 'derselben', 'des', 'deshalb', 'desselben', 'dessen', 'deswegen', 'd.h', 'dich', 'die', 'diejenige', 'diejenigen', 'dies', 'diese', 'dieselbe', 'dieselben', 'diesem', 'diesen', 'dieser', 'dieses', 'dir', 'doch', 'dort', 'drei', 'drin', 'dritte', 'dritten', 'dritter', 'drittes', 'du', 'durch', 'durchaus', 'dürfen', 'dürft', 'durfte', 'durften', 'e', 'eben', 'ebenso', 'ehrlich', 'ei', 'ei,', 'ei,', 'eigen', 'eigene', 'eigenen', 'eigener', 'eigenes', 'ein', 'einander', 'eine', 'einem', 'einen', 'einer', 'eines', 'einige', 'einigen', 'einiger', 'einiges', 'einmal', 'einmal', 'eins', 'elf', 'en', 'ende', 'endlich', 'entweder', 'entweder', 'er', 'Ernst', 'erst', 'erste', 'ersten', 'erster', 'erstes', 'es', 'etwa', 'etwas', 'euch', 'f', 'früher', 'fünf', 'fünfte', 'fünften', 'fünfter', 'fünftes', 'für', 'g', 'gab', 'ganz', 'ganze', 'ganzen', 'ganzer', 'ganzes', 'gar', 'gedurft', 'gegen', 'gegenüber', 'gehabt', 'gehen', 'geht', 'gekannt', 'gekonnt', 'gemacht', 'gemocht', 'gemusst', 'genug', 'gerade', 'gern', 'gesagt', 'gesagt', 'geschweige', 'gewesen', 'gewollt', 'geworden', 'gibt', 'ging', 'gleich', 'gott', 'gross', 'groß', 'grosse', 'große', 'grossen', 'großen', 'grosser', 'großer', 'grosses', 'großes', 'gut', 'gute', 'guter', 'gutes', 'h', 'habe', 'haben', 'habt', 'hast', 'hat', 'hatte', 'hätte', 'hatten', 'hätten', 'heisst', 'her', 'heute', 'hier', 'hin', 'hinter', 'hoch', 'i', 'ich', 'ihm', 'ihn', 'ihnen', 'ihr', 'ihre', 'ihrem', 'ihren', 'ihrer', 'ihres', 'im', 'im', 'immer', 'in', 'in', 'indem', 'infolgedessen', 'ins', 'irgend', 'ist', 'j', 'ja', 'ja', 'jahr', 'jahre', 'jahren', 'je', 'jede', 'jedem', 'jeden', 'jeder', 'jedermann', 'jedermanns', 'jedoch', 'jemand', 'jemandem', 'jemanden', 'jene', 'jenem', 'jenen', 'jener', 'jenes', 'jetzt', 'k', 'kam', 'kann', 'kannst', 'kaum', 'kein', 'keine', 'keinem', 'keinen', 'keiner', 'kleine', 'kleinen', 'kleiner', 'kleines', 'kommen', 'kommt', 'können', 'könnt', 'konnte', 'könnte', 'konnten', 'kurz', 'l', 'lang', 'lange', 'lange', 'leicht', 'leide', 'lieber', 'los', 'm', 'machen', 'macht', 'machte', 'mag', 'magst', 'mahn', 'man', 'manche', 'manchem', 'manchen', 'mancher', 'manches', 'mann', 'mehr', 'mein', 'meine', 'meinem', 'meinen', 'meiner', 'meines', 'mensch', 'menschen', 'mich', 'mir', 'mit', 'mittel', 'mochte', 'möchte', 'mochten', 'mögen', 'möglich', 'mögt', 'morgen', 'muss', 'muß', 'müssen', 'musst', 'müsst', 'musste', 'mussten', 'n', 'na', 'nach', 'nachdem', 'nahm', 'natürlich', 'neben', 'nein', 'neue', 'neuen', 'neun', 'neunte', 'neunten', 'neunter', 'neuntes', 'nicht', 'nicht', 'nichts', 'nie', 'niemand', 'niemandem', 'niemanden', 'noch', 'nun', 'nun', 'nur', 'o', 'ob', 'ob', 'oben', 'oder', 'oder', 'offen', 'oft', 'oft', 'ohne', 'Ordnung', 'p', 'q', 'r', 'recht', 'rechte', 'rechten', 'rechter', 'rechtes', 'richtig', 'rund', 's', 'sa', 'sache', 'sagt', 'sagte', 'sah', 'satt', 'schlecht', 'Schluss', 'schon', 'sechs', 'sechste', 'sechsten', 'sechster', 'sechstes', 'sehr', 'sei', 'sei', 'seid', 'seien', 'sein', 'seine', 'seinem', 'seinen', 'seiner', 'seines', 'seit', 'seitdem', 'selbst', 'selbst', 'sich', 'sie', 'sieben', 'siebente', 'siebenten', 'siebenter', 'siebentes', 'sind', 'so', 'solang', 'solche', 'solchem', 'solchen', 'solcher', 'solches', 'soll', 'sollen', 'sollte', 'sollten', 'sondern', 'sonst', 'sowie', 'später', 'statt', 't', 'tag', 'tage', 'tagen', 'tat', 'teil', 'tel', 'tritt', 'trotzdem', 'tun', 'u', 'über', 'überhaupt', 'übrigens', 'uhr', 'um', 'und', 'und?', 'uns', 'unser', 'unsere', 'unserer', 'unter', 'v', 'vergangenen', 'viel', 'viele', 'vielem', 'vielen', 'vielleicht', 'vier', 'vierte', 'vierten', 'vierter', 'viertes', 'vom', 'von', 'vor', 'w', 'wahr?', 'während', 'währenddem', 'währenddessen', 'wann', 'war', 'wäre', 'waren', 'wart', 'warum', 'was', 'wegen', 'weil', 'weit', 'weiter', 'weitere', 'weiteren', 'weiteres', 'welche', 'welchem', 'welchen', 'welcher', 'welches', 'wem', 'wen', 'wenig', 'wenig', 'wenige', 'weniger', 'weniges', 'wenigstens', 'wenn', 'wenn', 'wer', 'werde', 'werden', 'werdet', 'wessen', 'wie', 'wie', 'wieder', 'will', 'willst', 'wir', 'wird', 'wirklich', 'wirst', 'wo', 'wohl', 'wollen', 'wollt', 'wollte', 'wollten', 'worden', 'wurde', 'würde', 'wurden', 'würden', 'x', 'y', 'z', 'z.b', 'zehn', 'zehnte', 'zehnten', 'zehnter', 'zehntes', 'zeit', 'zu', 'zuerst', 'zugleich', 'zum', 'zum', 'zunächst', 'zur', 'zurück', 'zusammen', 'zwanzig', 'zwar', 'zwar', 'zwei', 'zweite', 'zweiten', 'zweiter', 'zweites', 'zwischen', 'zwölf']

    @staticmethod
    def text_to_wordlist(acticle):
        try:
            acticle_text = re.sub("[^a-zA-ZöÖüÜäÄß]"," ", acticle)
            acticle_text = re.sub("\s\s+"," ", acticle_text).lower()
            # acticle_text = re.sub(r'https?:\/\/.*[\r\n]*', '', acticle, flags=re.MULTILINE)
            # acticle_text = re.sub(r'<\/?em>', '', acticle_text, flags=re.MULTILINE)
        except:
            acticle_text = ''
        return acticle_text

    @staticmethod
    def to_wordlist(data):
        return data.apply(Helper.text_to_wordlist)

    @staticmethod
    def remove_stopwords(data):
        return data.apply(lambda x: [item for item in str(x).split(' ') if item not in Helper.stop])

    @staticmethod
    def merge(data):
        return data.apply(lambda x: " ".join(x))

    @staticmethod
    def stem(data):
        return data.apply(lambda x: " ".join([Helper.stemmer.stem(y) for y in x]))

    @staticmethod
    def remove_stop(data):
        data = Helper.to_wordlist(data)
        data = Helper.remove_stopwords(data)
        data = Helper.merge(data)
        return data

    @staticmethod
    def remove_stop_and_stem(data):
        data = Helper.to_wordlist(data)
        data = Helper.remove_stopwords(data)
        data = Helper.stem(data)
        return data
