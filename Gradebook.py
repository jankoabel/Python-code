last_semester_gradebook = [["politics", 80], ["latin", 96], ["dance", 97], ["architecture", 65]]
subjects = ["Linear Algebra", "Mathematics1", "Programming","Logical System","Human Machine Interface"]
scores = [99, 99, 99, 99, 99]
gradebook = [["Linear Algebra", 99], ["Mathematics1", 99], ["Programming",99], ["Logical System", 99],["Human Machine Interface", 99]]
print(gradebook)
gradebook.append(["computer Science", 100])
gradebook.append(["visual arts", 93])
gradebook[-1][-1]=(98)

print(gradebook)
gradebook[4].remove(99)
print(gradebook)
gradebook[4].append("Pass")
print(gradebook)
full_gradebook = last_semester_gradebook + gradebook
print(full_gradebook)
