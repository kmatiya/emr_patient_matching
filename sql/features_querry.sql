select mwp.patient_id, mwp.first_name, mwp.last_name,mwp.gender, mwp.birthdate,
mwp.district, mwp.traditional_authority, mai.clinical_stage, mai.tb_status,
mai.art_first_line_regimen, art_first_line_regimen_start_date, mai.pregnant_or_lactating,
mai.location as enrollment_location,mai.height as height_at_enrollment,mai.weight as weight_at_enrollment, 
first_encounter.first_visit_date,first_encounter.first_visit_location, first_encounter.regimen_at_first_visit, 
first_encounter.height_at_first_visit,first_encounter.weight_at_first_visit,last_encounter.last_visit_date, 
last_encounter.height_at_last_visit, last_encounter.weight_at_last_visit, last_encounter.regimen_at_last_visit, "upper_neno" as db
from mw_patient mwp 
join mw_art_initial mai
on mwp.patient_id = mai.patient_id
join (
	select x.patient_id, max(x.visit_date) as last_visit_date, 
    x.location as last_visit_location, x.art_regimen as regimen_at_last_visit, 
    x.height as height_at_last_visit, x.weight as weight_at_last_visit
    from mw_art_followup x
    join (
		select patient_id, max(visit_date) as visit_date
		from mw_art_followup
        where height is not null and art_regimen is not null
        group by patient_id
    ) y
    on x.patient_id = y.patient_id and x.visit_date = y.visit_date
    group by patient_id

) last_encounter
on mwp.patient_id = last_encounter.patient_id
join (
	select x.patient_id, min(x.visit_date) as first_visit_date, 
    x.location as first_visit_location, x.art_regimen as regimen_at_first_visit,
    x.height as height_at_first_visit, x.weight as weight_at_first_visit
    from mw_art_followup x
    join (
		select patient_id, min(visit_date) as visit_date
		from mw_art_followup
        where height is not null and art_regimen is not null
        group by patient_id
    ) y
    on x.patient_id = y.patient_id and x.visit_date = y.visit_date
    group by patient_id
) first_encounter 
on mwp.patient_id = first_encounter.patient_id
