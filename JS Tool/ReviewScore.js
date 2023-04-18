/**
 * takes a rawSheet and translate it to the target sheet
 * @param {*} params 
 * @param {*} rawSheet 
 * @param {*} source                Source language to translate from
 * @param {*} listOfLangs           Array of languages to translate ; ['en-US'] is necessary for calculating the ReviewScore
 * @param {*} calculateReview       wheather method should calculate the review score or not? (boolean)
 * @param {*} externalsTasks 
 * @param {*} google 
 * @param {*} auth 
 */
async function __translateAndEvaluateSheet(params, rawSheet, source, listOfLangs, externalsTasks, google, auth, num_translation = 1) {


    // Define the formality of the translation of Deepl
    // 0 = default, 1 = formal, -1 = informal
    const formality = 0;

    // Dumb required.. @TODO manage this ASAP !
    externalsTasks = externalsTasks;


    // Set the column name based on the given source
    const _baseSourceColumnName = getColName(source);
    // const _baseReferenceColumnName = `English [en-EN]`;
    const options = params.options || {};
    const shouldWriteDataSheet = options.shouldWrite ?? true; // Write on output by default... should be changed
    const calculateReview = options.shouldReview ?? false;
    const entryLimits = options.entryLimits ?? -1;
    const filters = options.filters ?? null;
    // Set to true to use ignore list sheet to avoid translating texts that are already translated
    const shouldUseIgnoreList = options.shouldUseIgnoreList ?? true;

    // set the source language
    const sourceLang = source;

    // Add new glossary
    // await externalsTasks.autoLocalizeTools.DeeplUpdateGlossary('fr', 'en', 'glossary/fr_en_glossary.csv');


    // Filter or get data
    let _filteredData = filters ? externalsTasks.updateSheetSource.getFilteredData(rawSheet, filters, __getElementValue) : rawSheet;


    if (entryLimits != -1) {
        _filteredData.length = entryLimits
    }


    // set source language and format the target language (As of today, Deepl doesn't support ISO language codes ex. fr-FR, but only for english it must contain en-US)
    var targetLangs = getTargetLanguages(listOfLangs);

    // Save the ignored list to a temp variable
    let _tempIgnoreList = [];
    let _ignoreList;

    if (shouldUseIgnoreList) {
        // Get Ignore list (list that doesn't need to be translated)
        _ignoreList = await getIgnoreList(externalsTasks, google, auth);

        // Keep only the entries that their target language is included in the targetLanguages variable
        _ignoreList = filterIgnoreList(_ignoreList, targetLangs);

        // Creates an array of ignore list with the id and the translation data
        _tempIgnoreList = getIgnoreListArray(_filteredData, _ignoreList);
    }

    // convert to HTML to keep the formatted data
    _filteredData = _filteredData.filter(entry => {
        // If there is an undefined text in the FRENCH text, remove it
        if (entry[_baseSourceColumnName] === undefined)
            return false;
        const newRowData = externalsTasks.autoLocalizeTools.convertToHTML(entry[_baseSourceColumnName]);
        entry[_baseSourceColumnName] = newRowData;
        return true;
    });

    // Object that holds all the translations
    let translatedArray = {};

    // Collect Text To translate in english
    // let outputArray = externalsTasks.googleSheetsTools.getRawSheetColumnValues(_filteredData, _baseSourceColumnName, externalsTasks);
    let outputArray = [];
    _filteredData.forEach(entry => {
        let _languageValue = entry[_baseSourceColumnName].formattedValue || ""; // manage empty cell with empty string

        // Pre process the data
        // Remove the white spaces at the beginning and the end of the string
        // NOTE: THIS METHOD IS NOT TESTED ENOUGH, TRIMMING MIGHT MESS UP THE FORMATTING SO IT'S NOT RECOMMENDED
        // NOTE: TO AVOID THIS METHOD, DO THE TRIMMING IN THE GOOGLE SHEET SOURCE DATA AND COMMENT THIS PART
        _languageValue = _languageValue.trim();
        // determine if _languageValue is string or not



        outputArray.push(_languageValue);
    })

    // Add Html tag for not translating Bracket contents
    outputArray = addNoTranslateHTMLTag(outputArray);

    // translate all languages
    // @TODo : faire une boucle tt les 500 ?
    let loopLength;

    // Should use the glossary of Lean
    const useGlossary = true;

    // Batch translation length
    loopLength = 250;

    // Translate using Deepl
    // translate each language from source language, if useGlossary is true and no glossary is found,
    // it will be translated automatically with no glossary
    // if number of translations is more than 1, then add a counter
    for (let i = 1; i <= num_translation; i++) {
        for (const tr of targetLangs) {
            let prefix = tr;
            if (i > 1)
                prefix = tr + '_' + i;
            translatedArray[prefix] = await streamDeeplRequests(outputArray, sourceLang.split('-')[0], tr, loopLength, useGlossary, _tempIgnoreList, formality);
        }
    }

    // Reduce object to value
    for (let key in translatedArray) {
        // translatedArray[key] = translatedArray[key].map(e => __removeNoTranslatePattern(e.text));
        translatedArray[key] = translatedArray[key].map(e => translationTools.removeNoTranslatePattern(e.text));
    }

    // convert from HTML and get the formatted data
    for (let key in translatedArray) {
        translatedArray[key].forEach((entry, n) => {
            const newVal = externalsTasks.autoLocalizeTools.convertFromHTML(entry);
            translatedArray[key][n] = newVal;
        });
    }

    // Min review score set to a big value to be updated in the loop
    // NOTE: UNCOMMENT THIS PART TO HAVE A MIN AND MAX REVIEW SCORE FOR EACH LANGUAGE
    // let minReviewScore = {};
    let minReviewScore = { val: 10000000 };

    // Max review score set to a small value to be updated in the loop
    // NOTE: UNCOMMENT THIS PART TO HAVE A MIN AND MAX REVIEW SCORE FOR EACH LANGUAGE
    // let maxReviewScore = {};
    let maxReviewScore = { val: -10000000 };

    for (let key in translatedArray) {

        // For each language, the min and max review score will be different, create a variable for each language
        // NOTE: UNCOMMENT THIS PART TO HAVE A MIN AND MAX REVIEW SCORE FOR EACH LANGUAGE
        // minReviewScore[key] = 10000000;
        // maxReviewScore[key] = -10000000;

        // Add value in map
        _filteredData.forEach((entry, n) => {
            // Get reference language value
            const colName = getColName(key);
            const _languageValue = __getElementValue(entry[colName]);

            // Get translated value of the object if exists. if that's not the case, it's already translated
            var _translatedValue;
            if (translatedArray[key].length > n)
                _translatedValue = translatedArray[key][n].formattedValue;
            else
                _translatedValue = _languageValue;

            // Add the translated value with the format [key]_translated, ex. en-US_translated
            entry[key + "_translated"] = __setElementValue({}, _translatedValue);
            entry[key + "_translated"]["formattedValue"] = _translatedValue;

            // Add formatting to the original item if it exists
            if (translatedArray[key].length > n) {
                if (translatedArray[key][n].hasOwnProperty('textFormatRuns'))
                    entry[key + "_translated"]['textFormatRuns'] = translatedArray[key][n].textFormatRuns;
            }

            // if cell custom format exists, copy it for other translations
            if (entry[_baseSourceColumnName].hasOwnProperty('userEnteredFormat')) {
                entry[key + "_translated"]['userEnteredFormat'] = entry[_baseSourceColumnName].userEnteredFormat;
            }

            // calculate review score
            // Review score can only be calculated if the translated language is
            // present in the original data (par exemple En-original et En-translated ou Fr-original et Fr-translated)
            if (calculateReview && _languageValue != undefined && _translatedValue != undefined && isLanguageExist(_baseSourceColumnName, _filteredData)) {
                calculateReviewScore(entry, _languageValue, _translatedValue, minReviewScore, maxReviewScore, _baseSourceColumnName, key, _tempIgnoreList);
            }
        });
    }

    // Get all keys from entry
    let entryKeys = Object.keys(_filteredData[0]);

    // Filter only keys that are reviewScores
    entryKeys = entryKeys.filter(element => element.includes("reviewScore"));

    // Produce finalData for each language
    entryKeys.forEach((rs, i) => {
        _filteredData.forEach((entry, n) => {
            if (entry[rs] !== undefined) {
                const customValue = entry[rs].userEnteredValue.numberValue;
                // NOTE: The normalization is based on the values in the list, so it's possible that it wouldn't be so accurate
                // Minimum will be the min review score in the list and max will be the max review score in the list
                // So a big original data is better to have accurate review scores
                // NOTE: UNCOMMENT THESE TWO LINES TO HAVE A MIN AND MAX REVIEW SCORE FOR EACH LANGUAGE
                // const key_name = rs.split('reviewScore_')[1];
                // entry[rs] = __setElementValue({}, normalize(customValue, minReviewScore[key_name], maxReviewScore[key_name]));
                entry[rs] = __setElementValue({}, normalize(customValue, minReviewScore.val, maxReviewScore.val));
            }
        });
    });

    // autoRecordSheetIfTarget Is Setted
    const containsTarget = params.target ?? false;
    if (shouldWriteDataSheet && containsTarget) {
        const _targetOutputSpreadSheetId = params.target.spreadSheetId;
        const _targetOutputSheetName = params.target.sheetName;
        const _headerRemap = null;
        await __writeObjectToGoogleSheet(_filteredData, _targetOutputSpreadSheetId, _targetOutputSheetName, externalsTasks, google, auth, _headerRemap);
    }


    return _filteredData;
}

/**
 * calculates review score for the given entry (for now only supports English)
 * @param {*} entry                     object to update with the new values
 * @param {*} _languageValue            original text value
 * @param {*} _translatedValue          translated text value
 * @param {*} minReviewScore            minimum review score
 * @param {*} maxReviewScore            maximum review score
 * @param {*} _baseSourceColumnName     column name of the source language
 * @param {*} _languageKey              ISO key of the source language
 * @param {*} _ignoreList               List of translation keys to ignore
 */
function calculateReviewScore(entry, _languageValue, _translatedValue, minReviewScore, maxReviewScore, _baseSourceColumnName, _languageKey, _ignoreList) {
    const distance = externalsTasks.autoLocalizeTools.Levenshtein_score(_languageValue, _translatedValue);
    const distanceB = externalsTasks.autoLocalizeTools.calculateBLEU(_languageValue, _translatedValue);
    const distanceTRI = externalsTasks.autoLocalizeTools.calculateTRI(_languageValue, _translatedValue);
    const distanceTER = externalsTasks.autoLocalizeTools.calculateTER(_languageValue, _translatedValue);

    entry[_languageKey + "_distance"] = __setElementValue({}, distance);
    entry[_languageKey + "_distance_blue"] = __setElementValue({}, distanceB);
    entry[_languageKey + "_distance_ter"] = __setElementValue({}, distanceTER);
    entry[_languageKey + "_distance_tri"] = __setElementValue({}, distanceTRI);
    entry[_languageKey + "_distance_LCS"] = __setElementValue({}, externalsTasks.autoLocalizeTools.longestCommonSubsequence(_languageValue, _translatedValue));

    /// ****************
    // Includes formula ?
    // @TODO : if includes formula, do not push ?
    let re = new RegExp(".*[0-9]*.*=.*[0-9]*.*");
    let re2 = new RegExp("(.*http(s)?.*)|(.*<[A-Za-z]+.*)");

    const includesFormula = re.test(_languageValue) && !re2.test(_languageValue);

    entry["includesFormula"] = __setElementValue({}, includesFormula);
    /// ****************

    /// ****************
    // Is it only numbers ? (including % and special characters)
    re = new RegExp("[A-Za-z]+");
    const justLetters = re.test(_languageValue);
    if (justLetters || _languageValue.trim() == "") {
        entry["onlyNumbers"] = __setElementValue({}, false);
    } else {
        entry["onlyNumbers"] = __setElementValue({}, true);
    }
    /// ****************

    /// ****************

    // Wheather or not this entry includes brackets or not
    // const includesBraces = __hasBrackets(_languageValue);
    const includesBraces = translationTools.hasBrackets(_languageValue);
    entry["includesBraces"] = __setElementValue({}, includesBraces);
    /// ****************

    /// ****************
    // Has formatting ?
    var hasFormatting = false;

    if (entry[_baseSourceColumnName].textFormatRuns != null && entry[_baseSourceColumnName].textFormatRuns.length > 0)
        hasFormatting = true;

    entry["hasFormatting"] = __setElementValue({}, hasFormatting);

    /// *******************
    // Calculate number of words
    const numberOfWords = __getNumberOfWords(_languageValue);
    entry["numberOfWords"] = __setElementValue({}, numberOfWords);
    /// *******************


    /// *******************
    // Check if contains link
    re = new RegExp("http(s)?:");
    var includesLink = re.test(_languageValue);
    entry["includesLink"] = __setElementValue({}, includesLink);

    /// *******************

    // Check if source is empty or MISSING
    let isSourceEmpty = false;
    if (entry[_baseSourceColumnName].userEnteredValue.stringValue.trim() === "" || entry[_baseSourceColumnName].userEnteredValue.stringValue === '{{MISSING}}')
        isSourceEmpty = true;

    /// *******************
    // Check if original language value is empty or missing
    let isOriginalEmpty = false;
    const originalColName = getColName(_languageKey);
    if (entry.hasOwnProperty(originalColName) && (entry[originalColName].userEnteredValue.stringValue.trim() === "" || entry[originalColName].userEnteredValue.stringValue === '{{MISSING}}'))
        isOriginalEmpty = true;

    /// *******************

    // Check if translation is empty or MISSING
    let isTranslationEmpty = false;
    if (_translatedValue.trim() === "" || _translatedValue === '{{MISSING}}')
        isTranslationEmpty = true;

    /// *******************

    let isInIgnoreList = false;

    _ignoreList.forEach((item, n) => {
        if (item.data['Key'].formattedValue === entry['Key'].formattedValue) {
            if (item.lang === _languageKey)
                isInIgnoreList = true;
        }
    });


    /// ********************
    // Now calculate the score of which user should review this cell

    let weightNumWords = 0.01;
    const weightHasFormatting = 0;
    const weightIncludesBraces = 0.2;
    const weightOnlyNumber = -10;
    const weightIncludesFormula = 2;
    let weightBleuScore = 21;
    let weightLevensteinScore = 5;
    const weightEmptySource = 42;
    const weightOriginalEmptySource = 42;
    const weightEmptyTranslation = 42;
    let weightInIgnoreList = -20;

    // if the BLEU score is one, it means both texts are the same; give more weight to the BLEU score
    if (distanceB == 1) {
        weightBleuScore *= -2;
        weightLevensteinScore = 0;
        weightNumWords = 0;
        weightInIgnoreList = 0;
    }

    let reviewScore = (1 / numberOfWords) * weightNumWords + hasFormatting * weightHasFormatting +
        includesBraces * weightIncludesBraces + !justLetters * weightOnlyNumber + includesFormula * weightIncludesFormula +
        distanceB * weightBleuScore + ((distance / 100)) * weightLevensteinScore + weightEmptySource * isSourceEmpty +
        weightInIgnoreList * isInIgnoreList + weightEmptyTranslation * isTranslationEmpty + weightOriginalEmptySource * isOriginalEmpty;


    // NOTE: UNCOMMENT THIS IF YOU WANT TO CALCULATE THE MAX AND MIN SCORES FOR EACH LANGUAGE
    // if (reviewScore > maxReviewScore[_languageKey])
    //     maxReviewScore[_languageKey] = reviewScore;

    // if (reviewScore < minReviewScore[_languageKey])
    //     minReviewScore[_languageKey] = reviewScore;

    if (reviewScore > maxReviewScore.val)
        maxReviewScore.val = reviewScore;

    if (reviewScore < minReviewScore.val)
        minReviewScore.val = reviewScore;

    entry["reviewScore_" + _languageKey] = __setElementValue({}, reviewScore);

    /// ******************
}
