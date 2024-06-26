# Generated by Django 4.2.13 on 2024-05-09 22:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("catalogo", "0001_initial"),
    ]

    operations = [
        migrations.AlterField(
            model_name="file",
            name="persons",
            field=models.ManyToManyField(
                blank=True, related_name="files", to="catalogo.person"
            ),
        ),
        migrations.AlterField(
            model_name="file",
            name="themes",
            field=models.ManyToManyField(
                blank=True, related_name="files", to="catalogo.theme"
            ),
        ),
    ]
