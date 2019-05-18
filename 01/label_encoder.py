from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
input_classes = ['audi', 'ford', 'audi', 'toyota', 'ford', 'bmw']
label_encoder.fit(input_classes)
print("\nClass mapping:")
for i, item in enumerate(label_encoder.classes_):
    print(item, '-->', i)
#OUT:Class mapping:
#audi --> 0
#bmw --> 1
#ford --> 2
#toyota --> 3

labels = ['toyota', 'ford', 'audi']
encoded_labels = label_encoder.transform(labels)
print("\nLabels =",labels)
print("Encoder labels =",list(encoded_labels))