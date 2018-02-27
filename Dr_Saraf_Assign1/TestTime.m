time_elapsed = 0;
for i=1:20
    time_elapsed = time_elapsed + Copy_of_Dr_Saraf_Car_Assignment(totalsulfurdioxide, freesulfurdioxide);
    disp(time_elapsed);
end
time_elapsed = time_elapsed/20;
disp([time_elapsed, ' seconds']);