$(function() {
    $('#destination').change(function() {
         var id = $(this).val();
         var data = "";
         $('#morris-bar-chart2').empty().append();
         $('#landing_1').show();
         $.ajax({
            type: 'GET',
            url:   '/graphe/'+id,
            dataType: 'json',
            async: true,
            contentType: "application/json; charset=utf-8",
            data: {},
            success: function (result) {
                 result.score = parseInt(result.score);
                 result.dates = new Date(result.dates);
                 Morris.Bar({
                    element: 'morris-bar-chart2',
                    data: result,
                    xkey: 'dates',
                    ykeys: ['score'],
                    labels: ['number'],
                    hideHover: 'auto',
                    resize: true,
                    });
                 $('#landing_1').hide();
            },
            error: function (xhr, status, error) {
                alert(error);
            }
        });

         return data;

    })

    $('#destination_boats').change(function() {
         var id = $(this).val();
         var data = "";
         $('#morris-bar-chart_destination_boats').empty().append();
         $('#landing_destination_boats').show();
         $.ajax({
            type: 'GET',
            url:   '/graphe_destination_boat/'+id,
            dataType: 'json',
            async: true,
            contentType: "application/json; charset=utf-8",
            data: {},
            success: function (result) {
                 result.score = parseInt(result.score);
                 Morris.Bar({
                    element: 'morris-bar-chart_destination_boats',
                    data: result,
                    xkey: ['name_boat'],
                    ykeys: ['score'],
                    labels: ['nombre request',],
                    hideHover: 'auto',
                    resize: true,
                    });
                 $('#landing_destination_boats').hide();
            },
            error: function (xhr, status, error) {
                alert(error);
            }
        });

         return data;

    })

    $('#boats').change(function() {
         var id = $(this).val();
         var data = "";
         $('#morris-bar-chart-boat').empty().append();
         $('#landing_boat').show();
         $.ajax({
            type: 'GET',
            url:   '/graphe_boat/'+id,
            dataType: 'json',
            async: true,
            contentType: "application/json; charset=utf-8",
            data: {},
            success: function (result) {

                 result.score = parseInt(result.score);
                 result.dates = new Date(result.dates);
                 $('#landing_boat').hide();
                 Morris.Bar({
                    element: 'morris-bar-chart-boat',
                    data: result,
                    xkey: 'dates',
                    ykeys: ['score'],
                    labels: ['number'],
                    hideHover: 'auto',
                    resize: true,
                    });

            },
            error: function (xhr, status, error) {
                alert(error);
                $('#landing_boat').hide();
            }
        });

         return data;

    })

    $('#country_boat').change(function() {
         var country = $(this).val();
         var data = "";
         $('#morris-bar-chart-country_boat').empty().append();
         $('#landing_country_boat').show();
         $.ajax({
            type: 'GET',
            url:   '/graphe_boat_country/'+country,
            dataType: 'json',
            async: true,
            contentType: "application/json; charset=utf-8",
            data: {},
            success: function (result) {
                 $('#landing_country_boat').hide();
                 Morris.Bar({
                    element: 'morris-bar-chart-country_boat',
                    data: result,
                    xkey: 'name',
                    ykeys: ['counts'],
                    labels: ['number'],
                    hideHover: 'auto',
                    resize: true,
                    });
                 $('#landing_country_boat').hide();
            },
            error: function (xhr, status, error) {
                alert(error);
                $('#landing_country_boat').hide();
            }
        });
         return data;
    })

    $('#year_request').change(function() {
         var year = $(this).val();
         var data = "";
         $('#morris-bar-chart_year').empty().append();
         $('#landing_year').show();
         $.ajax({
            type: 'GET',
            url:   '/graphe_year_request/'+year,
            dataType: 'json',
            async: true,
            contentType: "application/json; charset=utf-8",
            data: {},
            success: function (result) {
                 result.month = parseInt(result.month);
                 $('#landing_year').hide();
                 Morris.Bar({
                    element: 'morris-bar-chart_year',
                    data: result,
                    xkey: 'month',
                    ykeys: ['counts'],
                    labels: ['number'],
                    hideHover: 'auto',
                    resize: true,
                    });
                 $('#landing_year').hide();
            },
            error: function (xhr, status, error) {
                alert(error);
                $('#landing_year').hide();
            }
        });
         return data;
    })

    $('#all_year').click(function(){
       $('#morris-bar-chart_all_year').empty().append();
       $('#landing_all_year').show();
       $.ajax({
            type: 'GET',
            url:   '/graphe_allyear_request',
            dataType: 'json',
            async: true,
            contentType: "application/json; charset=utf-8",
            data: {},
            success: function (result) {
                 $('#landing_all_year').hide();
                 Morris.Bar({
                    element: 'morris-bar-chart_all_year',
                    data: result,
                    xkey: 'year',
                    ykeys: ['counts'],
                    labels: ['number'],
                    hideHover: 'auto',
                    resize: true,
                    });
                 $('#landing_all_year').hide();
            },
            error: function (xhr, status, error) {
                alert(error);
                $('#landing_all_year').hide();
            }
        });
         return data;
    })

    $('#country_alldestination').change(function() {
         var country = $(this).val();
         var data = "";
         $('#morris-bar-chart-country_alldestination').empty().append();
         $('#landing_country_alldestination').show();
         $.ajax({
            type: 'GET',
            url:   '/graphe_alldestination_country/'+country,
            dataType: 'json',
            async: true,
            contentType: "application/json; charset=utf-8",
            data: {},
            success: function (result) {
                 $('#landing_country_alldestination').hide();
                 Morris.Bar({
                    element: 'morris-bar-chart-country_alldestination',
                    data: result,
                    xkey: 'name',
                    ykeys: ['counts'],
                    labels: ['number'],
                    hideHover: 'auto',
                    resize: true,
                    });
                 $('#landing_country_alldestination').hide();
            },
            error: function (xhr, status, error) {
                alert(error);
                $('#landing_country_alldestination').hide();
            }
        });
         return data;
    })

});
$(function() {

    $('#boats_des').change(function() {
         var id_boat = $(this).val();
         var id_des = $('#des_boats').val();
         var data = "";
         $('#morris-bar-chart-boat-destination').empty().append();
         $('#landing_boat_destination').show();
         $.ajax({
            type: 'GET',
            url:   '/graphe_boat_des/'+id_boat+'/'+id_des,
            dataType: 'json',
            async: true,
            contentType: "application/json; charset=utf-8",
            data: {},
            success: function (result) {
                 result.score = parseInt(result.score);
                 result.dates = new Date(result.dates);
                 $('#landing_boat_destination').hide();
                 Morris.Bar({
                    element: 'morris-bar-chart-boat-destination',
                    data: result,
                    xkey: 'dates',
                    ykeys: ['score'],
                    labels: ['number'],
                    hideHover: 'auto',
                    resize: true,
                    });
                 $('#landing_boat_destination').hide();
            },
            error: function (xhr, status, error) {
                alert(error);
                $('#landing_boat_destination').hide();
            }
        });
         return data;
    })

    $('#des_boats').change(function() {
         var id_des  = $(this).val();
         var id_boat = $('#boats_des').val();
         var data = "";
         $('#morris-bar-chart-boat-destination').empty().append();
         $('#landing_boat_destination').show();
         $.ajax({
            type: 'GET',
            url:   '/graphe_boat_des/'+id_boat+'/'+id_des,
            dataType: 'json',
            async: true,
            contentType: "application/json; charset=utf-8",
            data: {},
            success: function (result) {
                 result.score = parseInt(result.score);
                 result.dates = new Date(result.dates);
                 $('#landing_boat_destination').hide();
                 Morris.Bar({
                    element: 'morris-bar-chart-boat-destination',
                    data: result,
                    xkey: 'dates',
                    ykeys: ['score'],
                    labels: ['number'],
                    hideHover: 'auto',
                    resize: true,
                    });
                 $('#landing_boat_destination').hide();
            },
            error: function (xhr, status, error) {
                alert(error);
                $('#landing_boat_destination').hide();
            }
        });
         return data;

    })


});
$(function() {

    $('#country_destination').change(function() {
         var country = $(this).val();
         var destination = $('#des_country').val();
         var data = "";
         $('#morris-bar-chart-country_destination').empty().append();
         $('#landing_country_destination').show();
         $.ajax({
            type: 'GET',
            url:   '/graphe_destination_country/'+country+'/'+destination,
            dataType: 'json',
            async: true,
            contentType: "application/json; charset=utf-8",
            data: {},
            success: function (result) {
                 result.dates = new Date(result.dates);
                 $('#landing_country_destination').hide();
                 Morris.Bar({
                    element: 'morris-bar-chart-country_destination',
                    data: result,
                    xkey: 'dates',
                    ykeys: ['counts'],
                    labels: ['number'],
                    hideHover: 'auto',
                    resize: true,
                    });
                 $('#landing_country_destination').hide();
            },
            error: function (xhr, status, error) {
                alert(error);
                $('#landing_country_destination').hide();
            }
        });
         return data;
    })

    $('#des_country').change(function() {
         var destination  = $(this).val();
         var country = $('#country_destination').val();
         var data = "";
         $('#morris-bar-chart-country_destination').empty().append();
         $('#landing_country_destination').show();
         $.ajax({
            type: 'GET',
            url:   '/graphe_destination_country/'+country+'/'+destination,
            dataType: 'json',
            async: true,
            contentType: "application/json; charset=utf-8",
            data: {},
            success: function (result) {
                 result.dates = new Date(result.dates);
                 $('#landing_boat_destination').hide();
                 Morris.Bar({
                    element: 'morris-bar-chart-country_destination',
                    data: result,
                    xkey: 'dates',
                    ykeys: ['counts'],
                    labels: ['number'],
                    hideHover: 'auto',
                    resize: true,
                    });
                 $('#landing_country_destination').hide();
            },
            error: function (xhr, status, error) {
                alert(error);
                $('#landing_country_destination').hide();
            }
        });
         return data;

    })


});
jQuery(function($){

    $('#landing_1').hide();
    $('#landing_2').hide();
    $('#landing_boat').hide();
    $('#landing_boat_destination').hide();
    $('#landing_destination_boat_type').hide();
    $('#landing_destination_boats').hide();
    $('#landing_country_boat').hide();
    $('#landing_country_destination').hide();
    $('#landing_year').hide();
    $('#landing_all_year').hide();
    $('#landing_country_alldestination').hide();


    $('#ajax_loading').hide();
    $('#ajax_loading_1').hide();
    $('#ajax_loading_srb').hide();
    $('#ajax_loading_sr_b_d').hide();
    $('#ajax_loading_data_boat').hide();
    $('#ajax_gear').hide();
    $('#ajax_gear_b_d').hide();
    $('#ajax_gear_boat').hide();


    $('#update_recommendation').bind('click',function(e){
        // Affichage du gif de chargement et envoi requête AJAX
         $('#ajax_loading').show();
         $('#refrech_rec_D').hide();
    });
    $('#update_data').bind('click',function(e){
        // Affichage du gif de chargement et envoi requête AJAX
         $('#ajax_loading_1').show();
         $('#refech_data_ml').hide();
    });
    $('#start_ml').bind('click',function(e){
        // Affichage du gif de chargement et envoi requête AJAX
         $('#ajax_gear').show();
    });
    $('#training_boat_destination').bind('click',function(e){
        // Affichage du gif de chargement et envoi requête AJAX
         $('#ajax_gear_b_d').show();
    });
    $('#training_ml_boat').bind('click',function(e){
        // Affichage du gif de chargement et envoi requête AJAX
         $('#ajax_gear_boat').show();
         $('#gear_ml_boat').hide();
    });
    $('#update_recommendation_b').bind('click',function(e){
        // Affichage du gif de chargement et envoi requête AJAX
         $('#ajax_loading_srb').show();
         $('#refrech_rec_b').hide();
    });
    $('#update_recommendation_boats_des').bind('click',function(e){
        // Affichage du gif de chargement et envoi requête AJAX
         $('#ajax_loading_sr_b_d').show();
         $('#refrech_rec_b_d').hide();
    });
    $('#update_data_boats').bind('click',function(e){
        // Affichage du gif de chargement et envoi requête AJAX
         $('#ajax_loading_data_boat').show();
         $('#refrech_data_boat').hide();
    });

});
$(function() {
    $('#country').change(function(){
         var id = $(this).val();
         var data = "";
         $('#morris-donut-chart').empty().append();
         $('#landing_2').show();
        $.ajax({
            type: 'GET',
            url:   '/type_boat_country/'+id,
            dataType: 'json',
            async: true,
            contentType: "application/json; charset=utf-8",
            data: {},
            success: function (result) {
                 result.value = parseInt(result.value);
                     Morris.Donut({
                            element: 'morris-donut-chart',
                            data: result,
                     resize: true
                });
                  $('#landing_2').hide();
              },
            error: function (xhr, status, error) {
                alert(error);
            }
        });
        return data;

    }) // Trigger the event

    $('#destination_boat_type').change(function(){
         var id = $(this).val();
         var data = "";
         $('#morris-donut-chart-destination_boat_type').empty().append();
         $('#landing_destination_boat_type').show();
         $.ajax({
            type: 'GET',
            url:   '/graphe_type_boat_des/'+id,
            dataType: 'json',
            async: true,
            contentType: "application/json; charset=utf-8",
            data: {},
            success: function (result) {
                 result.value = parseInt(result.value);
                     Morris.Donut({
                            element: 'morris-donut-chart-destination_boat_type',
                            data: result,
                     resize: true
                });
                $('#landing_destination_boat_type').hide();
            },
            error: function (xhr, status, error) {
                alert(error);
            }
        });
        return data;

    })
});