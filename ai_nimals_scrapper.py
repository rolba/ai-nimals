from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import os
import json
import urllib3
import sys
import time
import shutil

searched_test_array = [ "Parus major", 
                        "Poecile montanus", 
                        "Carduelis flammea",
                        "Parus cristatus",
                        "Carduelis spinus",
                        "Turdus iliacus", 
                        "Dryocopus martius",    
                        "Dendrocopos major",    
                        "Picus canus",
                        "Picus viridis",    
                        "Dendrocopos medius",    
                        "Dendrocopos minor",    #NOK
                        "Carduelis chloris",    
                        "Corvus frugilegus",
                        "Pyrrhula pyrrhula",    
                        "Columba livia f. urbana",    
                        "Coccothraustes coccothraustes",    
                        "Accipiter gentilis",    
                        "Bombycilla garrulus",    
                        "Fringilla montifringilla",    
                        "Corvus monedula",        
                        "Turdus merula",    
                        "Sitta europaea",    
                        "Accipiter nisus",    
                        "Corvus corax",
                        "Turdus pilaris",                
                        "Carduelis cannabina",        
                        "Passer montanus",    
                        "Larus canus",    
                        "Larus argentatus",    
                        "Parus caeruleus",    
                        "Regulus regulus",    
                        "Buteo buteo",                
                        "Certhia familiaris",    
                        "Certhia brachydactyla",
                        "Emberiza calandra", 
                        "Emberiza schoeniclus",    
                        "Aegithalos caudatus",        
                        "Erithacus rubicola",    
                        "Carduelis flavirostris",    
                        "Streptopelia decaocto",    
                        "Parus palustris",    
                        "Parus ater", 
                        "Pica pica",
                        "Lanius excubitor",  
                        "Troglodytes troglodytes",    
                        "Carduelis carduelis",    
                        "Sturnus vulgaris",    
                        "Garrulus glandarius",    
                        "Emberiza citrinella",    
                        "Corvus corone",    
                        "Passer domesticus",    
                        "Panurus biarmicus",
                        "Fringilla coelebs",    
                        "Larus ridibundus"]


num_requested = 1000

# adding path to geckodriver to the OS environment variable
os.environ["PATH"] += os.pathsep + os.getcwd()
download_path = "~/Downloads"

def main():
    print ("Scrapping started")
    if not os.path.exists(download_path):
        os.makedirs(download_path)
#     else:
#         shutil.rmtree(download_path)
#         os.makedirs(download_path)
        
    for searchtext in searched_test_array:

        number_of_scrolls = int((num_requested / 400) + 10) 
        # number_of_scrolls * 400 images will be opened in the browser
    
        if not os.path.exists(download_path + searchtext.replace(" ", "_")):
            os.makedirs(download_path + searchtext.replace(" ", "_"))
        else:
            shutil.rmtree(download_path + searchtext.replace(" ", "_"))
            os.makedirs(download_path + searchtext.replace(" ", "_"))
        url = "https://www.google.com/search?q="+searchtext+"&source=lnms&tbm=isch"
        driver = webdriver.Firefox()
        driver.get(url)
    
        headers = {}
        headers['User-Agent'] = "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36"
        extensions = {"jpg", "jpeg", "png", "gif"}
        img_count = 0
        downloaded_img_count = 0
        
        for _ in range(number_of_scrolls):
            for __ in range(10):
                # multiple scrolls needed to show all 400 images
                driver.execute_script("window.scrollBy(0, 1000000)")
                time.sleep(0.2)
            # to load next 400 images
            time.sleep(0.5)
            try:
                driver.find_element_by_xpath("//input[@value='Więcej wyników']").click()
                
            except Exception as e:
                print ("Less images found:", e)
                break
    
        imges = driver.find_elements_by_xpath('//div[contains(@class,"rg_meta")]')
        print ("Total images:", len(imges), "\n")
        for img in imges:
            img_count += 1
            img_url = json.loads(img.get_attribute('innerHTML'))["ou"]
            img_type = json.loads(img.get_attribute('innerHTML'))["ity"]
            print ("Downloading image", img_count, ": ", img_url, img_type)
            try:
                if img_type not in extensions:
                    img_type = "jpg"
    #             req = urllib3.request(img_url, headers=headers)
                http = urllib3.PoolManager()
                response = http.request('GET', img_url, timeout = 30)
    #             raw_img = urllib3.urlopen(req).read()
                f = open(download_path+searchtext.replace(" ", "_")+"/"+str(downloaded_img_count)+"."+img_type, "wb")
                f.write(response.data)
                f.close
                downloaded_img_count += 1
            except Exception as e:
                print ("Download failed:", e)
            finally:
                print
            if downloaded_img_count >= num_requested:
                break
    
        print ("Total downloaded: ", downloaded_img_count, "/", img_count)
        driver.quit()
        time.sleep(0.5)
    
    print ("Scrapping done")

if __name__ == "__main__":
    main()
