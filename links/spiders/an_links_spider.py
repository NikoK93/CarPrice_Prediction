import scrapy
from scrapy import FormRequest
from scrapy.spiders import CrawlSpider, Rule, Spider, Request
#from avto_net.items import AvtoNetItem
from scrapy.loader import ItemLoader
import csv

class AvtoSpider(Spider):


    name = 'avto'

    def start_requests(self):

        yield Request(
            'https://www.avto.net/Ads/search_makes.asp',
            callback=self.parse,
            meta={
                'handle_httpstatus_list': [302],
            },
        )

    def parse(self, response): 
        #Get's the links to every sub page you wan't to visit
        links = response.xpath("//a[@class='stretched-link font-weight-bold text-decoration-none text-truncate d-block']/@href").extract()
        #looping over the list of links
        print(links)
        for link in links:
            #for each link, we wan't to join it with the base link, since we're scraping the realtive path
            absolute_next_page_url = response.urljoin(link)
            #yielding the url and parsing in to parse_attr
            yield {
                'data':absolute_next_page_url
            }
