# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html


import scrapy
from scrapy.loader.processors import MapCompose, TakeFirst, Join, Identity
from w3lib.html import remove_tags, replace_escape_chars
import re
from unicodedata import normalize

def remove_quationms(value):
    return value.replace(u"\u201d", '').replace(u"\u201c", '')

def remove_char(value):
    return normalize('NFKD', value)



class AvtoNetItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    #images = scrapy.Field()
    #images_urls = scrapy.Field()
    url = scrapy.Field(
    input_processor = MapCompose(),
    output_processor = TakeFirst()
    )

    title = scrapy.Field(
    input_processor = MapCompose(),
    output_processor = Join()
    )
    price = scrapy.Field(
    input_processor = MapCompose(replace_escape_chars),
    output_processor = TakeFirst()
    )
    prodajalec = scrapy.Field(
    input_processor = MapCompose(replace_escape_chars),
    output_processor = Join()
    )
    features = scrapy.Field(
    input_processor = MapCompose(replace_escape_chars,remove_tags,remove_char),
    output_processor = Identity()
    )
    # registracija = scrapy.Field(
    # input_processor = MapCompose(replace_escape_chars),
    # output_processor = TakeFirst()
    # )
    # letnik = scrapy.Field(
    # input_processor = MapCompose(replace_escape_chars),
    # output_processor = TakeFirst()
    # )
    # starost = scrapy.Field(
    # input_processor = MapCompose(replace_escape_chars),
    # output_processor = TakeFirst()
    # )
    # teh_pregled = scrapy.Field(
    # input_processor = MapCompose(replace_escape_chars),
    # output_processor = TakeFirst()
    # )
    # prevozeni_km = scrapy.Field(
    # input_processor = MapCompose(replace_escape_chars),
    # output_processor = TakeFirst()
    # )
    # gorivo = scrapy.Field(
    # input_processor = MapCompose(replace_escape_chars),
    # output_processor = TakeFirst()
    # )
    # moto = scrapy.Field(
    # input_processor = MapCompose(replace_escape_chars),
    # output_processor = TakeFirst()
    # )
    # menjalnik = scrapy.Field(
    # input_processor = MapCompose(replace_escape_chars),
    # output_processor = TakeFirst()
    # )
    # oblika = scrapy.Field(
    # input_processor = MapCompose(replace_escape_chars),
    # output_processor = TakeFirst()
    # )
    # barva = scrapy.Field(
    # input_processor = MapCompose(replace_escape_chars),
    # output_processor = TakeFirst()
    # )
    # notranjost = scrapy.Field(
    # input_processor = MapCompose(replace_escape_chars),
    # output_processor = TakeFirst()
    # )
    # kraj_ogleda = scrapy.Field(
    # input_processor = MapCompose(replace_escape_chars),
    # output_processor = TakeFirst()
    # )
    # komb_voznja = scrapy.Field(
    # input_processor = MapCompose(replace_escape_chars),
    # output_processor = TakeFirst()
    # )
    # izvenmestna_voznja = scrapy.Field(
    # input_processor = MapCompose(replace_escape_chars),
    # output_processor = TakeFirst()
    # )
    # mestna_voznja = scrapy.Field(
    # input_processor = MapCompose(replace_escape_chars),
    # output_processor = TakeFirst()
    # )
    # emisijski_razred = scrapy.Field(
    # input_processor = MapCompose(replace_escape_chars),
    # output_processor = TakeFirst()
    # )
    # emisija_co2 = scrapy.Field(
    # input_processor = MapCompose(replace_escape_chars),
    # output_processor = TakeFirst()
    # )
    #ostalo = scrapy.Field(
    #input_processor = MapCompose(replace_escape_chars),
    #output_processor = Identity()
    
