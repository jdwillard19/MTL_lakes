library(sbtools)

##################################################################################
# (Jared - Sept 2020) - pull all data needed for MTL paper from sciencebase
# (note) - in the future should use queries if available instead of hard coding IDs
#################################################################################
cat("Enter ScienceBase username: ");
un <- readLines("stdin",n=1);
cat("Enter ScienceBase password: ");
pas <- readLines("stdin",n=1);
cat( "\n" )

authenticate_sb(un,pas)


dest_dir = '../../data/raw/sb_mtl_data_release/'


# pb0 predictions
pb0_item_str = c(
'5ebe569582ce476925e44b2f?f=__disk__a5%2F5b%2Fcd%2Fa55bcda87e4b93ce0379b5bc2a81321dcdfdf1b9',
'5ebe569582ce476925e44b2f?f=__disk__bd%2Fc5%2F59%2Fbdc559b96de439b1959aabae5abfa2d092b22375',
'5ebe569582ce476925e44b2f?f=__disk__b4%2Fc1%2Fb9%2Fb4c1b97338a0a35370de4058b089ecca23f52b9d',
'5ebe569582ce476925e44b2f?f=__disk__08%2F06%2Fae%2F0806aecf53437e1a9a0e2425b269884ad29c1f03',
'5ebe569582ce476925e44b2f?f=__disk__de%2F01%2Fb7%2Fde01b70d1557bdc078dfd0cf4f0af095ff2507ca',
'5ebe569582ce476925e44b2f?f=__disk__a3%2Fbd%2F1f%2Fa3bd1fcba0424e5cf53a8a178463a6b3e44b5338',
'5ebe569582ce476925e44b2f?f=__disk__99%2F40%2Fef%2F9940efdf3eacb0ddd7f82c5cdb2f54a78532bb8b',
'5ebe569582ce476925e44b2f?f=__disk__02%2Fd2%2F21%2F02d2213a1969acb31b1e9c6dcbf962b3e8a8160d',
'5ebe569582ce476925e44b2f?f=__disk__1d%2Fb5%2F59%2F1db5599a33c9b7666f6acf1c8ee8abfb61a85c9d',
'5ebe569582ce476925e44b2f?f=__disk__b5%2F6a%2F05%2Fb56a05bf57f18fb8ba0b33fa9aaba1662a7949fd',
'5ebe569582ce476925e44b2f?f=__disk__99%2F33%2F97%2F993397d04f8dfd35aeb78c7775f799db3ec22860',
'5ebe569582ce476925e44b2f?f=__disk__6c%2F1b%2Fda%2F6c1bda9917f380ae6f485fb53828ce688adcebae',
'5ebe569582ce476925e44b2f?f=__disk__94%2F74%2F0b%2F94740bc591d7ebcdb0680676f48067f43eec99ed',
'5ebe569582ce476925e44b2f?f=__disk__7e%2Fe5%2Ff1%2F7ee5f1a4bf7a524957f80de83f96c920e20ec430',
'5ebe569582ce476925e44b2f?f=__disk__3c%2Fac%2F1a%2F3cac1aaabc8ccc0c92c1a9f6a30b231227917717',
'5ebe569582ce476925e44b2f?f=__disk__ef%2Ffa%2F31%2Feffa3156011a1b7ca21d4368b09aeb6c622e3bf8')


for (item in pb0_item_str)
{
item_file_download(item,dest_dir=dest_dir) 
}




# model inputs 
item_file_download('5ebe568182ce476925e44b2d',overwrite_file=TRUE,dest_dir=dest_dir)

#lake metadata
item_file_download('5ebe564782ce476925e44b26?f=__disk__59%2Fe4%2Fac%2F59e4ac7164496cf60ad5db619349c9caf93e8152',overwrite_file=TRUE,dest_dir=dest_dir)


#temperature obs 
item_file_download('5ebe566a82ce476925e44b29?f=__disk__a5%2F63%2F23%2Fa56323d939ab21d71e160e355d304ac027567551',overwrite_file=TRUE,dest_dir=dest_dir)

# pb0 configs
item_file_download('5ebe567782ce476925e44b2b?f=__disk__87%2F71%2F17%2F877117acec395850457dfa317c0d576b453bcdda',overwrite_file=TRUE,dest_dir=dest_dir)


