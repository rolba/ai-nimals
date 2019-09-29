from selenium import webdriver
import os
import json
import urllib3
import time
import shutil

searched_test_array = [ "Parus major", 
                        "Poecile montanus",
                        "Carduelis flammea",
                        "Parus cristatus"]

num_requested = 1000

# adding path to geckodriver to the OS environment variable
os.environ["PATH"] += os.pathsep + os.getcwd()
download_path = os.getcwd() + "/Downloads"

def main():
    print ("Scrapping started")

    # Create Donwload patch or delete existing!
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    else:
        shutil.rmtree(download_path)
        os.makedirs(download_path)

    # Iterate over search array
    for searchtext in searched_test_array:

        # Create class patch of delete existing
        searchedTextDir = os.path.join(download_path, searchtext.replace(" ", "_"))
        if not os.path.exists(searchedTextDir):
            os.makedirs(searchedTextDir)
        else:
            shutil.rmtree(searchedTextDir)
            os.makedirs(searchedTextDir)

        # Prepare search URL. searchtext is a name of a class.
        url = "https://www.google.com/search?q="+searchtext+"&source=lnms&tbm=isch"
        # Start Firefox
        driver = webdriver.Firefox()
        # Open URL
        driver.get(url)

        extensions = {"jpg", "jpeg", "png", "gif"}
        img_count = 0
        downloaded_img_count = 0

        # I have to do some magic math to make web browser scroll down the search box.
        number_of_scrolls = int((num_requested / 400) + 10)
        for _ in range(number_of_scrolls):
            for __ in range(10):
                # And scroll scroll scroll to let Google Json load  images
                driver.execute_script("window.scrollBy(0, 1000000)")
                time.sleep(0.2)
            # to load next 400 images
            time.sleep(0.5)
            try:
                # Look for a button down the page for more search results.
                # For English version use: //input[@value='Show more results']
                driver.find_element_by_xpath("//input[@value='Więcej wyników']").click()
                
            except Exception as e:
                print ("Less images found:", e)
                break

        # Get URLs of all images on the page
        imges = driver.find_elements_by_xpath('//div[contains(@class,"rg_meta")]')
        print ("Total images:", len(imges), "\n")

        # Start iterating over found URLs
        for img in imges:
            img_count += 1
            img_url = json.loads(img.get_attribute('innerHTML'))["ou"]
            img_type = json.loads(img.get_attribute('innerHTML'))["ity"]
            print ("Downloading image", img_count, ": ", img_url, img_type)
            try:
                # Thy to save image on HDD
                if img_type not in extensions:
                    img_type = "jpg"
                http = urllib3.PoolManager()

                # Write image to hdd. Don't forget about timeout!
                response = http.request('GET', img_url, timeout = 10)
                f = open(searchedTextDir+"/"+str(downloaded_img_count)+"."+img_type, "wb")
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
