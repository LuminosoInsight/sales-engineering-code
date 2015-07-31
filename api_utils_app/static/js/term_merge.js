$(function() {
    //CODE FOR MERGE
    //CODE FOR TERM_UTILS_SEARCH STARTS HERE
    $('input[name=submitSearchMerge]').bind('click', function() {

        $("#loading1").show();
        var statusCode1 = '<h4>Loading terms...please wait</h4>';   
        $('#progress1').html(statusCode1);
        console.log("load bar started");

        console.log("sending over:" + $('input[name=acct]').val());
        console.log("sending over:" + $('input[name=proj]').val());
        console.log("sending over:" + $('input[name=query]').val());

      $.getJSON($SCRIPT_ROOT + '/term_utils/search', {
        acct: $('input[name=acct]').val(),
        proj: $('input[name=proj]').val(),
        query: $('input[name=query]').val()
      }, function(data) {

       $("#loading1").hide();
       $('#progress1').hide();
       console.log("load bar hidden");

       console.log("data submitted:");
       var search = JSON.stringify(data);
        //$('#result').text(w);

        var obj = JSON.parse(search);

        var selectTermsCode = '<h4>Select which terms you would like to merge:</h4>';
        selectTermsCode += '<form><select id="term_select1" multiple="multiple">';
        
        for(var key in obj){
          if (obj.hasOwnProperty(key)){
           var value=obj[key];
           //console.log("key:" + key);
           //console.log("value:" + value);
           selectTermsCode += '<option name="mergeTerms[]" value="'+value+'">'+key+'</option>';
          }
        }
        selectTermsCode += '</select>'
        selectTermsCode += '<br><br><input type="submit" class="btn btn-lg btn-primary btn-block" name="merge" value="Merge"/></form>'

        $('#selectTermsToMerge').html(selectTermsCode);

        $('#term_select1').multiselect({
          nonSelectedText: 'Select Terms to Merge',
          enableCaseInsensitiveFiltering: true,
          includeSelectAllOption: true,
          maxHeight: 400,
          buttonWidth: '300px',
        });

        $('input[name=merge]').bind('click', function() {
            $("#loading2").show();
            console.log("load bar started");
            var statusCode2 = '<p class="text-center"><h4>Recalculating...please wait</h4></p>';
            $('#progress2').html(statusCode2);

            var terms_checked = [];   
            $('option[name="mergeTerms[]"]:selected').each(function() {
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
            $("#progress2").hide();
            $("#loading2").hide();
            console.log("load bar hide");
            var completionCode = '<h4>Completed. Below are the merged terms</h4>';
            completionCode += '<table class="table">';
            var results =  JSON.stringify(data);
            var mergedTerms = JSON.parse(results);

            for(var key in mergedTerms){
              var list = mergedTerms[key];
              console.log("key:" + key);
              console.log("list:" + list);
              completionCode += '<tr>';
              completionCode += '<td>'+key+' '+list+'</td></tr>';
             }
            completionCode += '</table>';
            
            $('#MergeCompletedShow').html(completionCode);
            return false;
            }); //end of function(data) for Merge
            return false;
         });  //end of .bind('click') for input[name=merge]

        
        
       
      }); //end of function(data) for submitSearchMerge
  
        return false;
      }); 

  });  //end of function document



