# -*- coding: utf-8 -*-
import scrapy
from libai.items import LibaiItem
from scrapy.http import Request
class PoemSpider(scrapy.Spider):
    name = 'poems'
    allowed_domains = ['shicimingju.com']
    start_urls = [
        'https://www.shicimingju.com/chaxun/zuozhe/1.html']

    def parse(self, response):
        item = LibaiItem()
        # get方法等同于extract_first方法
        article = response.xpath(
            '//div[@class="shici_content"]//text()').getall()  # 等同于extract()方法
        item = LibaiItem(content=article)
        print(article)
        yield item
        next_page_part = response.xpath('//div[@id="list_nav_part"]/a/@href').getall()
        next_page = next_page_part[-2]
        print("下一页",next_page)
        yield Request("https://www.shicimingju.com" + next_page, callback=self.parse)
            
