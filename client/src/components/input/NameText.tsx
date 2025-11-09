import React from "react";

const NameText = ({
  userName,
  setUserName,
  sendUserName,
  text,
}: {
  userName: string;
  setUserName: (value: string) => void;
  sendUserName: () => void;
  text: string;
}) => {
  return (
    <div className="md:w-[440px] w-[360px] flex flex-col items-center mx-auto mt-10">
      <textarea
        placeholder="Enter the text"
        value={userName}
        onChange={(e) => setUserName(e.target.value)}
        className="w-full text-sm p-4 rounded-sm border border-[#E9EDD9] shadow-sm"
        rows={1}
      />
      <div
        onClick={sendUserName}
        className="w-full text-sm text-[#7E8766] p-4 rounded-sm mt-2 bg-[#F7F9F2] border border-[#E9EDD9] shadow-sm cursor-pointer text-center hover:bg-[#EEF2E0]"
      >
        {text}
      </div>
    </div>
  );
};

export default React.memo(NameText);
