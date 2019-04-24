import xml.etree.ElementTree as ET

tree = ET.parse('../../../datasets/sentiment_cornetto.xml')

objects = tree.findall('.//word')

word = []
id_ = []
sense = []
polarity = []
subjectivity = []
confidence = []
for object in objects:
    word.append(object.attrib["form"])
    id_.append(object.attrib["cornetto_id"])
    sense.append(object.attrib["sense"])
    polarity.append(object.attrib["polarity"])
    subjectivity.append(object.attrib["subjectivity"])
    confidence.append(object.attrib["confidence"])

emotion = pd.DataFrame(list(zip(word, id_, sense, polarity, subjectivity, confidence)), 
             columns=['word', 'id', 'sense', 'polarity','subjectivity','confidence'])

emotion.to_csv('emotion_cornetto.csv', index=False)