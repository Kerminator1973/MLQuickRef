import json
import pandas as pd

#tv_shows_json = pd.read_json('tv_shows.json')

with open("tv_shows.json", "r", encoding="utf-8") as f:
    tv_shows_json = json.load(f)

    print(type(tv_shows_json))

    normalized_df = pd.json_normalize(
        data = tv_shows_json["shows"],
        record_path = "episodes",
        meta=["show", "runtime", "network"])


    '''
    normalized_df = pd.json_normalize(
        tv_shows_json,
        record_path = ["shows", "episodes"],
        meta = [
            ["shows", "show"],
            ["shows", "runtime"],
            ["shows", "network"]
        ]
    )
    '''

    print(normalized_df.columns)

    xfiles_movies = normalized_df[normalized_df["show"] == "The X-Files"]
    buffy_movies = normalized_df[normalized_df["show"] == "Buffy the Vampire Slayer"]
    lost_movies = normalized_df[normalized_df["show"] == "Lost"]

    #print(xfiles_movies.head())
    #print(buffy_movies.head())
    #print(lost_movies.head())

    excel_file = pd.ExcelWriter("episodes.xlsx")

    xfiles_movies.to_excel(
        excel_file,
        sheet_name = "The X-files",
        index = False
    )

    buffy_movies.to_excel(
        excel_file,
        sheet_name = "Buffy the Vampire Slayer",
        index = False
    )

    lost_movies.to_excel(
        excel_file,
        sheet_name = "Lost",
        index = False
    )

    excel_file.close()
