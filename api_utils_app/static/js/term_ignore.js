 $(function() {
    //CODE FOR MERGE
    //CODE FOR TERM_UTILS_SEARCH STARTS HERE
    $('input[name=submitSearchIgnore]').bind('click', function() {
        console.log("sending over:" + $('input[name=acct_ignore]').val());
        console.log("sending over:" + $('input[name=proj_ignore]').val());
        console.log("sending over:" + $('input[name=query_ignore]').val());
      $.getJSON($SCRIPT_ROOT + '/term_utils/search', {
        acct: $('input[name=acct_ignore]').val(),
        proj: $('input[name=proj_ignore]').val(),
        query: $('input[name=query_ignore]').val()
      }, function(data) {
       console.log("data submitted:");
       var search = JSON.stringify(data);
        //$('#result').text(w);

        var obj = JSON.parse(search);

        var selectTermsCode = '<h4>Select which terms you would like to ignore:</h4>';
        selectTermsCode += '<form>';
        
        for(var key in obj){
          if (obj.hasOwnProperty(key)){
           var value=obj[key];
           //console.log("key:" + key);
           //console.log("value:" + value);
           selectTermsCode += '<input type="checkbox" name="ignoreTerms[]" value="'+value+'"/>'+key;
          }
        }
        selectTermsCode += '<input type="submit" class="btn btn-lg btn-primary btn-block" name="ignore" value="Ignore"/></form>'

        $('#selectTermsToIgnore').html(selectTermsCode);

        $('input[name=ignore]').bind('click', function() {
            var terms_checked = [];   
            $('input[name="ignoreTerms[]"]:checked').each(function() {
                 terms_checked.push($(this).val());
              });
            var terms_checkedStr = JSON.stringify(terms_checked);
            console.log("sending over:" + $('input[name=acct_ignore]').val());
            console.log("sending over:" + $('input[name=proj_ignore]').val());
            console.log("terms_checkedStr: " + terms_checkedStr);
            
            $.getJSON($SCRIPT_ROOT + '/term_utils/ignore', {
            acct: $('input[name=acct_ignore]').val(),
            proj: $('input[name=proj_ignore]').val(),
            terms: terms_checkedStr
            }, function(data) {
            console.log("data submitted, ignored");
    
            var statusCode = '<h4>Completed.</h4>';
            
            $('#IgnoreCompletedShow').html(statusCode);
            return false;
            }); //end of function(data) for Merge
            return false;
         });  //end of .bind('click') for input[name=merge]

        
        
       
      }); //end of function(data) for submitSearchMerge
  
        return false;
      }); 

  });  //end of function document


