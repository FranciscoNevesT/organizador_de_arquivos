from django.db import models

class Person(models.Model):
  numeric_id = models.AutoField(primary_key= True)
  name = models.CharField(max_length=100)
  surname = models.CharField(max_length=100)

  def __str__(self):
    return "{},{}".format(self.surname,self.name)

class Theme(models.Model):
  numeric_id = models.AutoField(primary_key=True)
  theme = models.CharField(max_length=100)

  def __str__(self):
    return self.theme

class File(models.Model):
  numeric_id = models.AutoField(primary_key= True)

  path = models.CharField(max_length=200,unique=True)

  name = models.CharField(max_length=200)
  size = models.IntegerField()
  type = models.CharField(max_length=10)
  creation_time = models.DateTimeField()
  modification_time = models.DateTimeField()

  persons = models.ManyToManyField(Person, related_name='files',blank=True)
  themes = models.ManyToManyField(Theme, related_name='files',blank=True)

  def __str__(self):
    return self.name

