 $(function() {
    //CODE FOR MERGE
    //CODE FOR TERM_UTILS_SEARCH STARTS HERE
    $('input[name=submitSearchIgnore]').bind('click', function() {
         $("#loading3").show();
          var statusCode3 = '<h4>Loading terms...please wait</h4>';   
          $('#progress3').html(statusCode3);
          console.log("load bar started");

        console.log("sending over:" + $('input[name=acct_ignore]').val());
        console.log("sending over:" + $('input[name=proj_ignore]').val());
        console.log("sending over:" + $('input[name=query_ignore]').val());
      $.getJSON($SCRIPT_ROOT + '/term_utils/search', {
        acct: $('input[name=acct_ignore]').val(),
        proj: $('input[name=proj_ignore]').val(),
        query: $('input[name=query_ignore]').val()
      }, function(data) {

        $("#loading3").hide();
        $('#progress3').hide();
        console.log("load bar hidden");

       console.log("data submitted:");
       var search = JSON.stringify(data);
        //$('#result').text(w);

        var obj = JSON.parse(search);

        var selectTermsCode = '<h4>Select which terms you would like to ignore:</h4>';
        selectTermsCode += '<form><select id="term_select2" multiple="multiple">';
        
        for(var key in obj){
          if (obj.hasOwnProperty(key)){
           var value=obj[key];
           //console.log("key:" + key);
           //console.log("value:" + value);
           selectTermsCode += '<option name="ignoreTerms[]" value="'+value+'">'+key+'</option>';
          }
        }
        selectTermsCode += '</select>'
        selectTermsCode += '<br><br><input type="submit" class="btn btn-lg btn-primary btn-block" name="ignore" value="Ignore"/></form>'

        $('#selectTermsToIgnore').html(selectTermsCode);

        $('#term_select2').multiselect({
          enableCaseInsensitiveFiltering: true,
          includeSelectAllOption: true,
          maxHeight: 400,
          buttonWidth: '300px',
        });

        $('input[name=ignore]').bind('click', function() {
            $("#loading4").show();
            var statusCode4 = '<h4>Recalculating...please wait</h4>';   
            $('#progress4').html(statusCode4);
            console.log("load bar started");

            var terms_checked = [];   
            $('option[name="ignoreTerms[]"]:selected').each(function() {
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
        
            $("#progress4").hide();
            $("#loading4").hide();

            console.log("data submitted, ignored");

            var completionCode = '<h4>Completed. Terms shown below are included in the ignore list</h4>';
          
            var results =  JSON.stringify(data); //{"fullstring":["wll|en","baen|en","yer|en","dx|en","bezos|en","www|en","yr|en","since|en"],"substring":[]}
            var obj = JSON.parse(results);
            completionCode += results+'<br>';

            
            $('#IgnoreCompletedShow').html(completionCode);
            return false;
            }); //end of function(data) for Merge
            return false;
         });  //end of .bind('click') for input[name=merge]

        
        
       
      }); //end of function(data) for submitSearchMerge
  
        return false;
      }); 

  });  //end of function document


