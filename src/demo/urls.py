from django.urls import path
from django.conf.urls import url
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('delete_files/', views.delete_files, name='delete_files'),
    path('generate_arxiv_query/', views.generate_arxiv_query, name='generate_arxiv_query'),
    path('get_survey_id/', views.get_survey_id, name='get_survey_id'),
    path('generate_pdf/', views.generate_pdf, name='generate_pdf'),
    path('save_outline/', views.save_outline, name='save_outline'),
    path('save_updated_cluster_info', views.save_updated_cluster_info, name='save_updated_cluster_info'),
    url(r'^get_topic$', views.get_topic, name='get_topic'),
    url(r'^get_survey$', views.get_survey, name='get_survey'),
    url(r'^automatic_taxonomy$', views.automatic_taxonomy, name='automatic_taxonomy'),
    url(r'^upload_refs$', views.upload_refs, name='upload_refs'),
    url(r'^annotate_categories$', views.annotate_categories, name='annotate_categories'),
    url(r'^select_sections$', views.select_sections, name='select_sections'),
    
]
