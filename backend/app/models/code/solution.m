%................................................................

function displacements=solution(GDof,prescribedDof,prescribedValue,stiffness,force)
% function to find solution in terms of global displacements
activeDof=setdiff([1:GDof]', ...
    [prescribedDof]);
force(activeDof)=-stiffness(activeDof,prescribedDof)*prescribedValue;
force(prescribedDof)=prescribedValue;

for i=1:length(prescribedDof)
    temp=stiffness(prescribedDof(i),prescribedDof(i));
    stiffness(prescribedDof(i),:)=0;
    stiffness(:,prescribedDof(i))=0;
    stiffness(prescribedDof(i),prescribedDof(i))=1;
end
U=stiffness\force;
displacements=zeros(GDof,1);
displacements=U;
