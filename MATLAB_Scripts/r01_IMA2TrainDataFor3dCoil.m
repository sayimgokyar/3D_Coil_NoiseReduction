%%Read the current directory, pick the different channels and acquisitions,
%and save them with their GT. Filenames should in theform of
%%Tag_.MR.SeqNumber_SliceNumber(InstanceNumber)_AcqNumber_CHX.IMA.
% For example:
% 3DCOIL_ACR_DRWANG_1.MR.0016.0001.ACQ0001_CH1.IMA
% 3DCOIL_ACR_DRWANG_1.MR.0016.0001.ACQ0001_CH2.IMA
% 3DCOIL_ACR_DRWANG_1.MR.0016.0001.ACQ0001_CH3.IMA
% 3DCOIL_ACR_DRWANG_1.MR.0016.0001.ACQ0001_GT.IMA

% 3DCOIL_ACR_DRWANG_1.MR.0016.0001.ACQ0002_CH1.IMA
% 3DCOIL_ACR_DRWANG_1.MR.0016.0001.ACQ0002_CH2.IMA
% 3DCOIL_ACR_DRWANG_1.MR.0016.0001.ACQ0002_CH3.IMA
% 3DCOIL_ACR_DRWANG_1.MR.0016.0001.ACQ0002_GT.IMA

%For iteration "n", #slice=300, #Acq=20, we will have 6000 slices/GT.
%%
clear; close all; clc;
current_path = cd();
input_path = '/Users/sayimgokyar/Desktop/3D_Coil/00_2D/20221220_Run13_JonathanWest/Processed';

cd(input_path);   A = dir();
%%
for ii=1:length(A)
    if startsWith(A(ii).name, 'COIL_3D_EXP13_1.MR')
        info = dicominfo(A(ii).name);
        tag = A(ii).name(1:12);

        disp([A(ii).name(1:12), ' ', info.Private_0051_100f]);

        if char(info.Private_0051_100f) == "A01"
            filename = [tag,'.SeriesNumber', num2str(info.SeriesNumber, '%03d'), '.Slice', num2str(info.InstanceNumber, '%03d'), '.Acq' ,num2str(info.AcquisitionNumber, '%03d'), '_CH1.IMA'];

            if isfile(filename)==0
                movefile(A(ii).name, filename);
            else
                disp('file already exists!');
            end

        elseif char(info.Private_0051_100f) == "A02"
            filename = [tag,'.SeriesNumber', num2str(info.SeriesNumber, '%03d'), '.Slice', num2str(info.InstanceNumber, '%03d'), '.Acq' ,num2str(info.AcquisitionNumber, '%03d'), '_CH2.IMA'];

            if isfile(filename)==0
                movefile(A(ii).name, filename);
            else
                disp('file already exists!');
            end

        elseif char(info.Private_0051_100f) == "A03"
            filename = [tag,'.SeriesNumber', num2str(info.SeriesNumber, '%03d'), '.Slice', num2str(info.InstanceNumber, '%03d'), '.Acq' ,num2str(info.AcquisitionNumber, '%03d'), '_CH3.IMA'];

            if isfile(filename)==0
                movefile(A(ii).name, filename);
                %If you have channels 1,2, and 3, then you need to clone/copy GT
                %from A(ii+1) with the current AcquisitionNumber and
                %InstanceNumber followed by _GT.IMA

%                 filename = [tag,'.SeriesNumber', num2str(info.SeriesNumber), '.Slice', num2str(info.InstanceNumber, '%03d'), '.Acq' ,num2str(info.AcquisitionNumber, '%03d'), '_GT.IMA'];
%                 info = dicominfo(A(ii+1).name); %Read the next files content before copying!
%                 
%                 if isfile(filename)==0 && char(info.Private_0051_100f) == "1_2,3" %Just make sure that next file is a GT file
%                         copyfile(A(ii+1).name, filename);
%                 else
%                     disp('Next file is not a GT file, despite coming after CH3 file!');
%                 end
            else
                disp('file already exists!');
            end

        elseif char(info.Private_0051_100f) == "1_2,3" %This is GT save it with a previous series number
            filename = [tag,'.SeriesNumber', num2str(info.SeriesNumber-1, '%03d'), '.Slice', num2str(info.InstanceNumber, '%03d'), '.Acq' ,num2str(info.AcquisitionNumber, '%03d'), '_GT.IMA'];

            if isfile(filename)==0
                movefile(A(ii).name, filename);
            else
                disp('file aready exists!');
            end
        else
            disp('not an appropriate image- deleting!');
            delete(A(ii).name);
        end
    end
end
cd(current_path);

