perl -pe's/([[:^ascii:]]|@|\^|\[|\]|[0-9]+|\{.*\}|\$)//g' $1 | java -jar beta_to_unicode.jar -s /dev/stdin -o /dev/stdout
