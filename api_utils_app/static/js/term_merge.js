$(function() {
    //CODE FOR MERGE
    //CODE FOR TERM_UTILS_SEARCH STARTS HERE
    $('input[name=submitSearchMerge]').bind('click', function() {
        console.log("sending over:" + $('input[name=acct]').val());
        console.log("sending over:" + $('input[name=proj]').val());
        console.log("sending over:" + $('input[name=query]').val());
      $.getJSON($SCRIPT_ROOT + '/term_utils/search', {
        acct: $('input[name=acct]').val(),
        proj: $('input[name=proj]').val(),
        query: $('input[name=query]').val()
      }, function(data) {
       console.log("data submitted:");
       var search = JSON.stringify(data);
        //$('#result').text(w);

        var obj = JSON.parse(search);

        var selectTermsCode = '<h4>Select which terms you would like to merge:</h4>';
        selectTermsCode += '<form>';
        
        for(var key in obj){
          if (obj.hasOwnProperty(key)){
           var value=obj[key];
           //console.log("key:" + key);
           //console.log("value:" + value);
           selectTermsCode += '<input type="checkbox" name="mergeTerms[]" value="'+value+'"/>'+key;
          }
        }
        selectTermsCode += '<input type="submit" class="btn btn-lg btn-primary btn-block" name="merge" value="Merge"/></form>'

        $('#selectTermsToMerge').html(selectTermsCode);

        $('input[name=merge]').bind('click', function() {
            var terms_checked = [];   
            $('input[name="mergeTerms[]"]:checked').each(function() {
                 terms_checked.push($(this).val());
              });
            var terms_checkedStr = JSON.stringify(terms_checked);
            console.log("sending over:" + $('input[name=acct]').val());
            console.log("sending over:" + $('input[name=proj]').val());
            console.log("terms_checkedStr: " + terms_checkedStr);
            
            $.getJSON($SCRIPT_ROOT + '/term_utils/merge', {
            acct: $('input[name=acct]').val(),
            proj: $('input[name=proj]').val(),
            terms: terms_checkedStr
            }, function(data) {
            console.log("data submitted, merged");
    
            var statusCode = '<h4>Completed.</h4>';
            
            $('#MergeCompletedShow').html(statusCode);
            return false;
            }); //end of function(data) for Merge
            return false;
         });  //end of .bind('click') for input[name=merge]

        
        
       
      }); //end of function(data) for submitSearchMerge
  
        return false;
      }); 

  });  //end of function document



