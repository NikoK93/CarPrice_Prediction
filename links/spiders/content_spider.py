import scrapy
from scrapy import FormRequest
from scrapy import loader
from scrapy.spiders import CrawlSpider, Rule, Spider, Request
from links.items import AvtoNetItem
from scrapy.loader import ItemLoader
import csv

class AvtoSpider(Spider):
    #handle_httpstatus_list = [301, 302, 307]
    name = 'avto_content'

    with open('url.txt') as file:
        start_urls = [line.strip() for line in file]

    def parse(self, response): 
        #Get's the links to every sub page you want to visit
        links = response.xpath("//a[@class='stretched-link']/@href").extract()
        #looping over the list of links
        print(links)
        for link in links:
            #for each link, we wan't to join it with the base link, since we're scraping the realtive path
            absolute_next_page_url = response.urljoin(link)
            print(absolute_next_page_url)
            
            #yielding the url and parsing in to parse_attr
            yield Request(url=absolute_next_page_url, callback=self.parse_attr)

        try:
            next_page_url = response.xpath('//*[@class="page-item GO-Rounded-R"]/a/@href').extract_first()
            abs_next_page_url = response.urljoin(next_page_url)
            yield Request(url=abs_next_page_url, callback=self.parse)
        except:
            print("no url")
        '''
        else:
            next_page_url = response.xpath('//div[@class="ResultsAdNavi123 Rounded"]/a[2]/@href').extract_first()
            abs_next_page_url = response.urljoin(next_page_url)
            yield Request(url=abs_next_page_url, callback=self.parse)
        '''

    #This function takes the sub url as input and scrapes all the relevant information for us. 
    def parse_attr(self, response):
            loader = ItemLoader(item=AvtoNetItem(), response=response)
            loader.add_value('url', response.url)
            loader.add_xpath('title',"//h3/text()")
            loader.add_xpath('price',"//p[@class='h2 font-weight-bold align-middle py-4 mb-0']/text()")
            loader.add_xpath('prodajalec',"//ul[@class='list-group list-group-flush bg-white p-0 pb-1 GO-Rounded-B text-center']/li/text()")
            loader.add_xpath('features',"//table[@class='table table-sm']/tbody/tr")
            #loader.add_xpath('ostalo',"(//div[@class='OglasEQRightWrapper'])/div/text()")
            yield loader.load_item()
